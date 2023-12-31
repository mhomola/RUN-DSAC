import gtimer as gt                                             # Timing of code execution
from collections import OrderedDict                             # A dictionary that remembers the order of key insertion
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
import rlkit.torch.pytorch_util as tool
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from .utils import LinearSchedule


def quantile_regression_loss(predict: torch.Tensor, target: torch.Tensor, tau: torch.Tensor, weight: torch.Tensor):

    """
    args:
        input:  (N, T)  - Input (predicted) quantile values
        target: (N, T)  - Target quantile values
        tau:    (N, T)  - Quantiles
    """

    predict                         = predict.unsqueeze(-1)
    target                          = target.detach().unsqueeze(-2)
    tau                             = tau.detach().unsqueeze(-1)
    weight                          = weight.detach().unsqueeze(-2)
    expanded_input, expanded_target = torch.broadcast_tensors(predict, target)        # Dimensions matching

    # Calculate Huber loss, delta = kappa
    L                               = F.huber_loss(expanded_input, expanded_target, reduction='none', delta=1.0)  # (N, T, T)
    sign                            = torch.sign(expanded_input - expanded_target) / 2. + 0.5
    rho                             = torch.abs(tau - sign) * L * weight
    loss                            = rho.sum(dim=-1).mean()                                                # Shape 1

    return loss

    #MaNotes: Identical

class DSACTrainer(TorchTrainer):

    def __init__(
            self,
            env,
            policy,
            target_policy,
            zf1,
            zf2,
            target_zf1,
            target_zf2,
            fp                      = None,
            target_fp               = None,
            discount                = 0.99,
            reward_scale            = 1.0,
            alpha                   = 1.0,
            policy_lr               = 3e-4,
            zf_lr                   = 3e-4,
            tau_type                = 'iqn',
            fp_lr                   = 1e-5,
            num_quantiles           = 32,
            optimizer_class         = optim.Adam,
            polyak_step             = 5e-3,
            target_update_period    = 1,
            clip_norm               = 1,    # Switch to 1 to clip gradients
            use_automatic_entropy_tuning = True,
            target_entropy          = -3,
            CAPS_std                = 0.05,  # Conditioning for Action Policy Smoothness (CAPS) (http://ai.bu.edu/caps/)
            lambda_t                = 0,
            lambda_s                = 0,
            std_init                = 0,
            std_final               = 0,
            lr_final                = 0.0002,
            schedule_length         = 0
    ):

        super().__init__()
        self.env                    = env
        self.policy                 = policy
        self.target_policy          = target_policy
        self.zf1                    = zf1
        self.zf2                    = zf2
        self.target_zf1             = target_zf1
        self.target_zf2             = target_zf2
        self.polyak_step            = polyak_step
        self.target_update_period   = target_update_period
        self.tau_type               = tau_type
        self.num_quantiles          = num_quantiles
        self.lambda_t               = lambda_t
        self.lambda_s               = lambda_s
        self.CAPS_std               = CAPS_std
        self.var_scale              = std_init
        self.var_scale_final        = std_final
        self.var_scale_schedule     = LinearSchedule(schedule_length, std_init, std_init if std_final is None else std_final)
        self.zf_lr_schedule         = LinearSchedule(100, zf_lr, zf_lr if lr_final is None else lr_final)
        self.policy_lr_schedule     = LinearSchedule(100, policy_lr, policy_lr if lr_final is None else lr_final)

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha          = tool.zeros(1, requires_grad = True)
            self.alpha_optimizer    = optimizer_class([self.log_alpha], lr=policy_lr)
        else:
            self.alpha = alpha

        self.zf_criterion       = quantile_regression_loss

        self.policy_optimizer   = optimizer_class(self.policy.parameters(), lr = policy_lr)
        self.zf1_optimizer      = optimizer_class(self.zf1.parameters(),    lr = zf_lr)
        self.zf2_optimizer      = optimizer_class(self.zf2.parameters(),    lr = zf_lr)

        self.fp                 = fp
        self.target_fp          = target_fp

        if self.tau_type == 'fqf':
            self.fp_optimizer = optimizer_class(self.fp.parameters(), lr = fp_lr)

        self.discount       = discount
        self.reward_scale   = reward_scale
        self.clip_norm      = clip_norm


        self.eval_statistics                    = OrderedDict()
        self._n_train_steps_total               = 0
        self.epoch_num                          = 0
        self._need_to_update_eval_statistics    = True

        self.avg_returns                        = []

    def get_tau(self, obs, actions, fp=None):

        """
        This function computes the quantile fractions (tau) based on the given observations (obs) and actions. The tau
        is calculated differently depending on the 'tau_type', which can be either 'iqn' (Implicit Quantile Networks)
        or 'fqf' (Fully parameterized Quantile Function). It also computes the midpoints of these fractions (tau_hat)

        Parameters:
        obs : array-like
            An array of observations.
        actions : array-like
            An array of actions based on the observations.
        fp : function, optional
            A function that takes in the observations and actions and returns the quantile fractions. This is used when tau_type is 'fqf'

        Returns:
        tau : torch.Tensor
            A tensor of quantile fractions based on the given observations and actions. These are the cumulative sums
            of the fractions calculated based on tau_type.
        tau_hat : torch.Tensor
            A tensor that represents the midpoints of these quantile fractions. It is computed by averaging adjacent
            fractions in tau.
        presum_tau : torch.Tensor
            A tensor of the calculated quantile fractions before they are summed cumulatively to produce tau

        Notes:
        'iqn' method generates a random set of quantile fractions and normalizes them such that they sum to 1.
        'fqf' method uses a user-provided function to compute the quantile fractions.
        """

        presum_tau  = None

        if self.tau_type == 'iqn':  # add 0.1 to prevent tau getting too close
            presum_tau  = tool.rand(len(actions), self.num_quantiles) + 0.1
            presum_tau /= presum_tau.sum(dim = -1, keepdims=True)
        elif self.tau_type == 'fqf':
            if fp is None:
                fp = self.fp
            presum_tau = fp(obs, actions)

        tau = torch.cumsum(presum_tau, dim = 1)  # (N, T), note that they are tau1...tauN in the paper

        with torch.no_grad():
            tau_hat = tool.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:]  = (tau[:, 1:] + tau[:, :-1]) / 2.

        return tau, tau_hat, presum_tau

    def train_from_torch(self, batch):
        rewards     = batch['rewards']
        terminals   = batch['terminals']
        obs         = batch['observations']
        actions     = batch['actions']
        next_obs    = batch['next_observations']
        gt.stamp('preback_start', unique=False)

        """
        Update Alpha - the adaptive temperature
        """
        new_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs,
            reparameterize  = True,
            return_log_prob = True,
        )

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss  = 0
            alpha       = self.alpha
        gt.stamp('preback_alpha', unique = False)

        """
        Update ZF
        """
        with torch.no_grad():
            new_next_actions, _, _, new_log_pi, *_ = self.target_policy(
                next_obs,
                reparameterize  = True,
                return_log_prob = True,
            )
            next_tau, next_tau_hat, next_presum_tau = self.get_tau(next_obs, new_next_actions, fp = self.target_fp)
            target_z1_values    = self.target_zf1(next_obs, new_next_actions, next_tau_hat)
            target_z2_values    = self.target_zf2(next_obs, new_next_actions, next_tau_hat)
            target_z_values     = torch.min(target_z1_values, target_z2_values) - alpha * new_log_pi
            z_target            = self.reward_scale * rewards + (1. - terminals) * self.discount * target_z_values

        tau, tau_hat, presum_tau = self.get_tau(obs, actions, fp=self.fp)
        z1_pred     = self.zf1(obs, actions, tau_hat)
        z2_pred     = self.zf2(obs, actions, tau_hat)
        zf1_loss    = self.zf_criterion(z1_pred, z_target, tau_hat, next_presum_tau)
        zf2_loss    = self.zf_criterion(z2_pred, z_target, tau_hat, next_presum_tau)
        gt.stamp('preback_zf', unique=False)

        # Optimize Z1
        self.zf1_optimizer.zero_grad()
        zf1_loss.backward()
        self.zf1_optimizer.step()
        gt.stamp('backward_zf1', unique=False)

        # Optimize Z2
        self.zf2_optimizer.zero_grad()
        zf2_loss.backward()
        self.zf2_optimizer.step()
        gt.stamp('backward_zf2', unique=False)

        """
        Update FP
        """
        if self.tau_type == 'fqf':
            with torch.no_grad():
                dWdtau = 0.5 * (2 * self.zf1(obs, actions, tau[:, :-1]) - z1_pred[:, :-1] - z1_pred[:, 1:] +
                                2 * self.zf2(obs, actions, tau[:, :-1]) - z2_pred[:, :-1] - z2_pred[:, 1:])
                dWdtau /= dWdtau.shape[0]  # (N, T-1)
            gt.stamp('preback_fp', unique=False)

            self.fp_optimizer.zero_grad()
            tau[:, :-1].backward(gradient=dWdtau)
            self.fp_optimizer.step()
            gt.stamp('backward_fp', unique=False)

        """
        Update Policy
        """

        with torch.no_grad():
            #print('new_actions', new_actions.shape)
            new_tau, new_tau_hat, new_presum_tau = self.get_tau(obs, new_actions, fp = self.fp)

        z1_new_actions = self.zf1(obs, new_actions, new_tau_hat)
        z2_new_actions = self.zf2(obs, new_actions, new_tau_hat)

        # Average values
        q1_new_actions = torch.sum(new_presum_tau * z1_new_actions, dim = 1, keepdims = True)
        q2_new_actions = torch.sum(new_presum_tau * z2_new_actions, dim = 1, keepdims = True)

        # Temporal smoothness loss term
        temp_smooth     = torch.mean((new_actions - new_next_actions).pow(2))
        L_temp_smooth   = self.lambda_t * temp_smooth / new_actions.shape[0]

        # Spatial smoothness loss term
        determ_action   = self.policy(obs)[1]
        nearby_action   = self.policy(torch.normal(mean=obs, std=self.CAPS_std))[1]
        space_smooth    = torch.mean((determ_action - nearby_action).pow(2))
        L_space_smooth  = self.lambda_s * space_smooth / new_actions.shape[0]

        L_CAPS          = L_temp_smooth + L_space_smooth

        # Variance parameter
        self.var_scale = self.var_scale_schedule(self.epoch_num)

        # Actions corrected for the standard deviation
        q1_new_actions, q2_new_actions = self.compute_qf_variance(self.var_scale, z1_new_actions, z2_new_actions,
                                                                  q1_new_actions, q2_new_actions, new_presum_tau)
        q_new_actions = torch.min(q1_new_actions, q2_new_actions)

        # Calculate policy loss

        policy_loss = (alpha * log_pi + L_CAPS - q_new_actions).mean()
        gt.stamp('preback_policy', unique=False)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad = tool.fast_clip_grad_norm(self.policy.parameters(), self.clip_norm)
        self.policy_optimizer.step()
        gt.stamp('backward_policy', unique=False)

        """
        Soft Updates
        """

        if self._n_train_steps_total % self.target_update_period == 0:

            tool.polyak_update(self.policy, self.target_policy, self.polyak_step)
            tool.polyak_update(self.zf1, self.target_zf1, self.polyak_step)
            tool.polyak_update(self.zf2, self.target_zf2, self.polyak_step)
            if self.tau_type == 'fqf':
                tool.polyak_update(self.fp, self.target_fp, self.polyak_step)

        """
        Save some statistics for eval
        """

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['ZF1 Loss']    = zf1_loss.item()
            self.eval_statistics['ZF2 Loss']    = zf2_loss.item()
            self.eval_statistics['Policy Loss'] = policy_loss.item()
            self.eval_statistics['Policy Grad'] = policy_grad
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z1 Predictions',
                tool.get_numpy(z1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z2 Predictions',
                tool.get_numpy(z2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z Targets',
                tool.get_numpy(z_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                tool.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                tool.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                tool.get_numpy(policy_log_std),
            ))

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha']       = alpha.item()
                self.eval_statistics['Alpha Loss']  = alpha_loss.item()

            print("Analysis of the QF:")
            print("Tau:", tau_hat[0])
            print("z1_pred:", z1_pred[0])
            print("Var param:", self.var_scale)
            print("lambda t:", self.lambda_t)
            print("lambda s:", self.lambda_s)
            print("Epoch #:", self.epoch_num)

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        """
        Increase epoch counter and initiate logging
        """
        self.epoch_num += 1
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):

        networks = [
            self.policy,
            self.target_policy,
            self.zf1,
            self.zf2,
            self.target_zf1,
            self.target_zf2,
        ]
        if self.tau_type == 'fqf':
            networks += [
                self.fp,
                self.target_fp,
            ]
        return networks

    def get_snapshot(self):

        snapshot = dict(
            policy          = self.policy.state_dict(),
            target_policy   = self.target_policy.state_dict(),
            zf1             = self.zf1.state_dict(),
            zf2             = self.zf2.state_dict(),
            target_zf1      = self.target_zf1.state_dict(),
            target_zf2      = self.target_zf2.state_dict(),
        )
        if self.tau_type == 'fqf':
            snapshot['fp']          = self.fp.state_dict()
            snapshot['target_fp']   = self.target_fp.state_dict()
        return snapshot

    def compute_qf_variance(self, factor, z1, z2, q1_expect, q2_expect, del_tau):

        # Variance calculation
        q1_var          = (del_tau * (z1 - q1_expect).pow(2)).sum(dim=1, keepdims=True)
        q2_var          = (del_tau * (z2 - q2_expect).pow(2)).sum(dim=1, keepdims=True)

        # Addition of standard deviation
        q1_new_actions  = q1_expect + factor * q1_var.sqrt()
        q2_new_actions  = q2_expect + factor * q2_var.sqrt()


        return q1_new_actions, q2_new_actions

