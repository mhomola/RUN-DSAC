import abc
import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import (
    PathCollector,
    StepCollector,
)


class OnlineRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: StepCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_paths_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch       = 1,
            min_num_steps_before_training   = 0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size                     = batch_size
        self.max_path_length                = max_path_length
        self.num_epochs                     = num_epochs
        self.num_eval_paths_per_epoch       = num_eval_paths_per_epoch
        self.num_trains_per_train_loop      = num_trains_per_train_loop
        self.num_train_loops_per_epoch      = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop  = num_expl_steps_per_train_loop
        self.min_num_steps_before_training  = min_num_steps_before_training

        assert self.num_trains_per_train_loop >= self.num_expl_steps_per_train_loop, \
            'Online training presumes num_trains_per_train_loop >= num_expl_steps_per_train_loop'

    def _train(self):
        self.training_mode(False)

        if self.min_num_steps_before_training > 0:
            # Collect new exploration steps according to given parameters
            self.expl_data_collector.collect_new_steps(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths = False,
            )

            # Get the initial exploration paths and add them to the replay buffer
            init_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(init_expl_paths)

            # Mark the end of the epoch for exploration data collector
            self.expl_data_collector.end_epoch(-1)

            gt.stamp('initial exploration', unique=True)

        # Compute the number of training iterations per exploration step
        num_trains_per_expl_step = self.num_trains_per_train_loop // self.num_expl_steps_per_train_loop

        # Iterate through epochs
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            # Collect new evaluation paths
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_paths_per_epoch,
            )
            gt.stamp('evaluation sampling')

            # Iterate through training loops per epoch
            for _ in range(self.num_train_loops_per_epoch):
                # Iterate through exploration steps per training loop
                for _ in range(self.num_expl_steps_per_train_loop):
                    # Collect new exploration steps
                    self.expl_data_collector.collect_new_steps(
                        self.max_path_length,
                        1,  # num steps
                        discard_incomplete_paths=False,
                    )
                    gt.stamp('exploration sampling', unique=False)

                    # Switch to training mode
                    self.training_mode(True)
                    # Iterate through training steps per exploration step
                    for _ in range(num_trains_per_expl_step):
                        # Fetch a random batch from the replay buffer and train
                        train_data = self.replay_buffer.random_batch(
                            self.batch_size)
                        self.trainer.train(train_data)
                    gt.stamp('training', unique=False)
                    self.training_mode(False)

            # Get new exploration paths and add them to the replay buffer
            new_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp('data storing', unique=False)

            self._end_epoch(epoch)
