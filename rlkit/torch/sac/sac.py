import gtimer as gt
from collections import OrderedDict
import numpy as np
import torch
import wandb
from torch import optim
from torch import nn
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
import tqdm


class SACTrainer(TorchTrainer):

    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            num_epochs,
            discount                    = 0.99,
            reward_scale                = 1.0,
            alpha                       = 0.2,
            policy_lr                   = 4.4e-4,
            qf_lr                       = 4.4e-4,
            optimizer_class             = optim.Adam,
            polyak_step                 = 5e-3,
            target_update_period        = 1,
            clip_norm                   = 1,
            use_automatic_entropy_tuning = True,
            target_entropy              = -3,
            CAPS_std                    = 0.05,
            lambda_t                    = 400,
            lambda_s                    = 400
    ):
        super().__init__()
        self.env                        = env
        self.policy                     = policy
        self.qf1                        = qf1
        self.qf2                        = qf2
        self.target_qf1                 = target_qf1
        self.target_qf2                 = target_qf2
        self.soft_target_tau            = polyak_step
        self.target_update_period       = target_update_period
        self.lambda_t                   = lambda_t
        self.lambda_s                   = lambda_s
        self.CAPS_std                   = CAPS_std
        self.policy_lr0                 = policy_lr
        self.qf_lr0                     = qf_lr
        self.policy_lr                  = policy_lr
        self.qf_lr                      = qf_lr
        self.num_epochs                 = num_epochs

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
                # -3 for the case of Attitude control
            self.log_alpha          = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer    = optimizer_class([self.log_alpha], lr = self.policy_lr)
        else:
            self.alpha              = alpha

        self.qf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr = self.policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr = self.qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr = self.qf_lr,
        )

        self.discount                           = discount
        self.reward_scale                       = reward_scale
        self.clip_norm                          = clip_norm
        self.eval_statistics                    = OrderedDict()
        self._n_train_steps_total               = 0
        self.epoch_num                          = 0
        self._need_to_update_eval_statistics    = True

        self.epoch_bar = tqdm.tqdm(
            desc            = "Current Episode",
            total           = self.num_epochs,
            unit            = "evaluation",
            position        = 2,
            colour          = "green",
            leave           = False,
        )

    def train_from_torch(self, batch):

        rewards     = batch['rewards']
        terminals   = batch['terminals']
        obs         = batch['observations']
        actions     = batch['actions']
        next_obs    = batch['next_observations']
        gt.stamp('preback_start', unique=False)

        """
        Update Alpha
        """
        new_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs,
            reparameterize  = True,
            return_log_prob = True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
            self.alpha = alpha
        else:
            alpha_loss  = 0
            alpha       = self.alpha
        gt.stamp('preback_alpha', unique = False)

        """
        Update QF
        """
        with torch.no_grad():
            new_next_actions, _, _, new_log_pi, *_ = self.policy(
                next_obs,
                reparameterize  = True,
                return_log_prob = True,
            )
            target_q_values = torch.min(
                self.target_qf1(next_obs, new_next_actions),
                self.target_qf2(next_obs, new_next_actions),
            ) - alpha * new_log_pi

            q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values

        q1_pred     = self.qf1(obs, actions)
        q2_pred     = self.qf2(obs, actions)
        qf1_loss    = self.qf_criterion(q1_pred, q_target)
        qf2_loss    = self.qf_criterion(q2_pred, q_target)
        gt.stamp('preback_qf', unique=False)

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()
        gt.stamp('backward_qf1', unique=False)

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()
        gt.stamp('backward_qf2', unique=False)

        """
        Update Policy
        """
        q_new_actions = torch.min(self.qf1(obs, new_actions), self.qf2(obs, new_actions))

        # Temporal smoothness loss term
        temp_smooth     = torch.mean((new_actions - new_next_actions).pow(2))
        L_temp_smooth   = self.lambda_t * temp_smooth / new_actions.shape[0]

        # Spatial smoothness loss term
        determ_action   = self.policy(obs)[1]
        nearby_action   = self.policy(torch.normal(mean=obs, std=self.CAPS_std))[1]
        space_smooth    = torch.mean((determ_action - nearby_action).pow(2))
        L_space_smooth  = self.lambda_s * space_smooth / new_actions.shape[0]

        L_CAPS = L_temp_smooth + L_space_smooth

        policy_loss = (alpha * log_pi + L_CAPS - q_new_actions).mean()
        gt.stamp('preback_policy', unique=False)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad = ptu.fast_clip_grad_norm(self.policy.parameters(), self.clip_norm)
        self.policy_optimizer.step()
        gt.stamp('backward_policy', unique=False)

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.polyak_update(self.qf1, self.target_qf1, self.soft_target_tau)
            ptu.polyak_update(self.qf2, self.target_qf2, self.soft_target_tau)

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            self.eval_statistics['QF1 Loss'] = qf1_loss.item()
            self.eval_statistics['QF2 Loss'] = qf2_loss.item()
            self.eval_statistics['Policy Loss'] = policy_loss.item()
            self.eval_statistics['Policy Grad'] = policy_grad
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

            print("lambdas:", self.lambda_t)
            print("Epoch #:", self.epoch_num)
            #print("Alpha:", self.alpha.item())

        self._n_train_steps_total   += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self.epoch_num += 1
        self._need_to_update_eval_statistics = True
        self.epoch_bar.update(1)

        self.policy_lr  = self.policy_lr0 - ((self.policy_lr0 - 0.00005) / (self.num_epochs)) * self.epoch_num
        self.qf_lr      = self.policy_lr0 - ((self.policy_lr0 - 0.00005) / (self.num_epochs)) * self.epoch_num
        self.policy_lr  = max(0.0, self.policy_lr)
        self.qf_lr      = max(0.0, self.qf_lr)

        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = self.policy_lr
        for param_group in self.qf1_optimizer.param_groups:
            param_group['lr'] = self.qf_lr
        for param_group in self.qf2_optimizer.param_groups:
            param_group['lr'] = self.qf_lr


    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy      = self.policy.state_dict(),
            qf1         = self.qf1.state_dict(),
            qf2         = self.qf2.state_dict(),
            target_qf1  = self.qf1.state_dict(),
            target_qf2  = self.qf2.state_dict(),
        )
