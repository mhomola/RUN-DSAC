from abc import ABC, abstractmethod
from typing import List, Tuple
import gym
import numpy as np
import tasks
from gym.spaces import Box
from tasks.random_signals import randomized_cosine_step_sequence, smoothed_step_sequence
from tools.utils import d2r, r2d
import wandb

def print_in_red(skk): print(f"\033[91m {skk}\033[00m")
def print_in_green(skk): print(f"\033[92m {skk}\033[00m")
def print_in_light_purple(skk): print(f"\033[94m {skk}\033[00m")
def print_in_purple(skk): print(f"\033[95m {skk}\033[00m")
def print_in_cyan(skk): print(f"\033[96m {skk}\033[00m")
def print_yellow(skk): print(f"\033[93m {skk}\033[00m")


class BaseEnv(gym.Env, ABC):
    """
    Base class to be able to write generic training & eval code that applies to all Citation environment variations.
    """

    def __init__(self, dt: float = 0.01):

        # Initialize variables
        self.dt                 = dt
        self.reference          = None
        self.track_list         = []
        self.states_list        = []
        self.action_list        = []
        self.error_list         = []
        self.reference_list     = []
        self.time_list          = []
        self.p_list             = []
        self.reward_list        = []

    @property
    @abstractmethod
    def action_space(self) -> Box:
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_space(self) -> Box:
        raise NotImplementedError

    @abstractmethod
    def calc_reference_value(self) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def get_controlled_state(self) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def calc_error(self) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def get_reward(self) -> float:
        pass

    def unscale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [action_space.low, action_space.high] to [-1, 1]

        Args:
            action (mp.ndarray): Action in the physical limits

        Returns:
            np.ndarray: Action vector in the [-1,1] interval for learning tasks
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def scale_action(self, clipped_action: np.ndarray) -> np.ndarray:
        """ Scale the action from [-1, 1] to [action_space.low, action_space.high].
        Might not be needed always since it depends on the activation of the output layer.

        Args:
            clipped_action (np.ndarray): Clipped action vector (deflections outputed by actor)

        Returns:
            np.ndarray: action vector in the physical limits
        """
        low, high = self.action_space.low, self.action_space.high
        return low + 0.5 * (clipped_action + 1.0) * (high - low)


class CitationEnv(BaseEnv):
    """ Citation wrapper. """

    # DASMAT inputs:
    #   0:  de      , 1:    da      , 2:    dr
    #   3:  trim de , 4:    trim da , 5:    trim dr
    #   6:  df      , 7:    gear    , 8:    throttle1, 9:   throttle2
    n_actions_full: int = 10            #

    # DASMAT states:
    #   0:  p   , 1:    q   ,   2:   r,
    #   3:  V   , 4:    alpha,  5:   beta,
    #   6:  phi , 7:    theta,  8:   psi,
    #   9:  he  ,10:    xe  ,  11:   ye
    n_obs_full:     int = 12            #

    t:              float = 0.          # [s]
    dt:             np.float16 = 0.01   # [s]

    def __init__(self, configuration, mode: str = "nominal"):

        super().__init__()
        print('This is configuration version', configuration["version"])
        self.gust = False

        # Select the configuration to be used
        if configuration["version"] == 'symmetric':
            print('Symmetric control only.\n')
            self.n_actions = 1                # theta
            self.obs_idx = [1]                # q
        elif configuration['version'] == 'attitude':
            print('Attitude control.\n')
            self.n_actions = 3                  # de, da, dr
            self.obs_idx = [0, 1, 2, 4]         # p,  q,  r, alpha
        else:
            print('Full state control.\n')
            self.n_actions = 3                  # de, da, dr
            self.obs_idx = range(10)            # all states

        if 'nominal' == mode or 'h2000-v90' in mode.lower() or 'incremental' in mode.lower():
            from .h2000_v90 import citation as citation_h2000_v90
            self.citation = citation_h2000_v90
            print_in_green('Trim mode: h=2000 m  v=90 m/s (nominal)')

        elif 'h2000-v150' == mode.lower() or 'high-q' == mode.lower():
            from .h2000_v150 import citation as citation_h2000_v150
            print_in_cyan('Trim mode: h=2000 m  v=150 m/s (high q)')
            self.citation = citation_h2000_v150

        elif 'h10000-v90' == mode.lower() or 'low-q' == mode.lower():
            from .h10000_v90 import citation as citation_h10000_v90
            self.citation = citation_h10000_v90
            print_in_cyan('Trim mode: h=10000 m  v=90 m/s (low q)')

        elif 'be' == mode.lower():
            from .be import citation as local_citation
            print_in_red('Trim mode: h=2000 m  v=90 m/s -- Broken Elevator (70%)')
            self.citation = local_citation

        elif 'jr' == mode.lower():
            from .jr import citation as local_citation
            print_in_red('Trim mode: h=2000 m  v=90 m/s -- Jammed Rudder at 15 deg')
            self.citation = local_citation

        elif 'se' == mode.lower():
            from .se import citation as local_citation
            print_in_red(
                'Trim mode: h=2000 m  v=90 m/s -- Saturated Elevator at +/-2.5 deg')
            self.citation = local_citation

        elif 'sa' == mode.lower():
            from .sa import citation as local_citation
            print_in_red(
                'Trim mode: h=2000 m  v=90 m/s -- Saturated Aileron at +/-1.0 deg')
            self.citation = local_citation

        elif 'ice' == mode.lower():
            from .ice import citation as citation_ice
            self.citation = citation_ice
            print_in_red('Trim mode: h=2000 m  v=90 m/s -- Iced')

        elif 'noise' == mode.lower():
            from .noise import citation as citation_noise
            self.citation = citation_noise
            print_in_red('Trim mode: h=2000 m  v=90 m/s -- Noisy Sensors')

        elif 'cg-for' == mode.lower():
            from .cg_for import citation as local_citation
            self.citation = local_citation
            print_in_red('Trim mode: h=2000 m  v=90 m/s --  CG Forward')

        elif 'cg' == mode.lower() or 'cg-aft' == mode.lower():
            from .cg import citation as local_citation
            self.citation = local_citation
            print_in_red('Trim mode: h=2000 m  v=90 m/s --  CG Aft')

        elif 'cg-shift' == mode.lower() or 'cg-timed' == mode.lower():
            from .cg_timed import citation as local_citation
            self.citation = local_citation
            print_in_red('Trim mode: h=2000 m  v=90 m/s --  CG Aft after 20s')

        elif 'gust' == mode.lower():
            from .gust import citation as local_citation
            self.citation = local_citation
            print_in_red(
                'Trim mode: h=2000 m  v=90 m/s --  Vertical Gust of 15ft/s at 20s')
            self.gust = True

        elif 'test' in mode.lower():
            from .test import citation as citation_test
            self.citation = citation_test
            print_in_red('TEST CASE')
        else:
            raise ValueError('Unknown trim condition or control mode!')

        # Use incremental control
        self.use_incremental = 'incremental' in mode.lower()
        if self.use_incremental:
            print('Incremental control.')

        # Configuration
        self.configuration      = configuration

        # Evaluation mode:
        self.eval_mode: bool    = False
        self.t_max              = configuration["training"]["t_max"]   # [s]

        # DASMAT state
        # 0,  1, 2   -> p, q, r
        # 3,  4, 5   -> V, alpha, beta
        # 6,  7, 8   -> phi, theta, psi
        # 9, 10, 11  -> he, xe, ye
        self.x: np.ndarray          = None    # observed state vector
        self.obs: np.ndarray        = None
        self.last_obs: np.ndarray   = None
        self.V0: float              = None  # m/s

        # DASMAT Inputs:
        #   0 de      , 1 da      , 2 dr
        #   3 trim de , 4 trim da , 5 trim dr
        #   6 df      , 7 gear    , 8 throttle1 9 throttle2
        self.last_u: np.ndarray     = None

        # reference to track
        self.ref: List[tasks.BaseSignal]    = None
        self.ref_values: np.ndarray         = None
        self.theta_trim: float              = 0.22     # standard theta trim deg

        # actuator bounds
        if self.use_incremental:
            self.bound = np.deg2rad(25)  # [deg/s]
        else:
            self.bound = np.deg2rad(10)  # [deg]

        # state bounds
        if self.configuration["aspiration"] == "eval" or self.configuration["aspiration"] == "eval q":
            self.max_theta  = 99999
            self.max_phi    = 99999
        else:
            self.max_theta = np.deg2rad(60.)
            self.max_phi   = np.deg2rad(75.)

        # observation space:
        if self.use_incremental:
            # aircraft state + actuator state + control states error (equal size withactuator states)
            self.n_obs: int = len(self.obs_idx) + 2 * self.n_actions
        else:
            # aircraft state + control states error
            self.n_obs: int = len(self.obs_idx) + self.n_actions

        # error
        self.error: np.ndarray = np.zeros((self.n_actions))

        # reward stuff
        if self.n_actions == 1:
            # individual reward scaler [theta]
            self.error_scaler = 6/np.pi * np.array([1.])
        else:
            # scaler [theta, phi, beta]
            self.error_scaler = 6/np.pi * np.array([1., 1., 4.])    # np.array([2., 1., 4.])
        self.error_scaler = self.error_scaler[:self.n_actions]
        self.max_bound = np.ones(self.error.shape)                  # bounds

        self.t_fin: float = 0.

    @property
    def action_space(self) -> Box:
        return Box(
            low     = np.deg2rad(np.array([-17.0, -19.0, -22.0])),  # np.deg2rad(np.array([-17.0, -17.0, -22.0]))
            high    = np.deg2rad(np.array([15.0, 15.0, 22.0])),     # np.deg2rad(np.array([17.0, 17.0, 22.0])),
            dtype   = np.float64,
        )

    @property
    def observation_space(self) -> Box:
        return Box(
            low     = -100 * np.ones(self.n_obs),                   # -30 * np.ones(self.n_obs),
            high    = 100 * np.ones(self.n_obs),                    # 30 * np.ones(self.n_obs),
            dtype   = np.float64,
        )

    @property
    def p(self):  # roll rate
        return self.x[0]

    @property
    def q(self):
        return self.x[1]

    @property
    def r(self):
        return self.x[2]

    @property
    def V(self) -> float:
        return self.x[3]

    @property
    def alpha(self) -> float:
        return self.x[4]

    @property
    def beta(self) -> float:
        return self.x[5]

    @property
    def phi(self) -> float:
        return self.x[6]

    @property
    def theta(self) -> float:
        return self.x[7]

    @property
    def psi(self) -> float:
        return self.x[8]

    @property
    def H(self) -> float:
        return self.x[9]

    @property
    def x_e(self) -> float:
        return self.x[10]

    @property
    def y_e(self) -> float:
        return self.x[11]

    @property
    def nz(self) -> float:
        """ Load factor [g] """
        return 1 + self.V * self.q/9.80665

    def set_eval_mode(self, t_max: int = 80) -> None:
        self.t_max = int(t_max)

        # if user_eval_refs is not None: self.user_refs = user_eval_refs
        self.eval_mode = True
        print_yellow(f'Switch to evaluation mode:\n Tmax = {self.t_max} s \n')

    def init_ref(self, **kwargs):
        if self.n_actions == 1:
            ref = randomized_cosine_step_sequence(
                t_max           = self.t_max,
                ampl_max        = 30,
                block_width     = self.t_max//5,
                smooth_width    = self.t_max//6.7,
                n_levels        = self.t_max//2,
                vary_timings    = self.t_max//500) + tasks.Const(self.theta_trim) #(0., self.t_max, self.theta_trim)
            self.ref = [ref]

        elif self.n_actions == 3:

            # Reference signal for Beta
            step_beta   = tasks.Const(value=0.0) #, self.t_max, 0.0)
            #step_theta  = tasks.Const(value=0.0)

            # Reference signals for theta and phi
            #if 'user_refs' not in kwargs:
            if self.configuration["aspiration"] == "train":
                self.theta_trim = np.rad2deg(self.theta) - 0.0006


                step_theta = randomized_cosine_step_sequence(
                    t_max           = self.t_max,
                    ampl_max        = self.configuration["pitch_ref"]["amplitude"], # 30,
                    block_width     = self.configuration["pitch_ref"]["block_w"],   # self.t_max//5,
                    smooth_width    = self.configuration["pitch_ref"]["smooth_w"],  # self.t_max // 6,
                    n_levels        = self.configuration["pitch_ref"]["n_levels"],  # self.t_max//2,
                    vary_timings    = self.configuration["pitch_ref"]["var_w"],     # self.t_max / 500.)
                    start_with_zero = self.configuration["pitch_ref"]["zero_start"])

                step_phi = randomized_cosine_step_sequence(
                    t_max           = self.t_max,
                    ampl_max        = self.configuration["roll_ref"]["amplitude"],   # 20,
                    block_width     = self.configuration["roll_ref"]["block_w"],     # self.t_max//5,
                    smooth_width    = self.configuration["roll_ref"]["smooth_w"],    # self.t_max//6,
                    n_levels        = self.configuration["roll_ref"]["n_levels"],    # self.t_max//2,
                    vary_timings    = self.configuration["roll_ref"]["var_w"],       # self.t_max / 500.)
                    start_with_zero = self.configuration["roll_ref"]["zero_start"])

            elif self.configuration["aspiration"] == "train_theta":
                self.theta_trim = np.rad2deg(self.theta) - 0.0006

                step_theta = randomized_cosine_step_sequence(
                    t_max           = self.t_max,
                    ampl_max        = self.configuration["pitch_ref"]["amplitude"], # 30,
                    block_width     = self.configuration["pitch_ref"]["block_w"],   # self.t_max//5,
                    smooth_width    = self.configuration["pitch_ref"]["smooth_w"],  # self.t_max // 6,
                    n_levels        = self.configuration["pitch_ref"]["n_levels"],  # self.t_max//2,
                    vary_timings    = self.configuration["pitch_ref"]["var_w"],     # self.t_max / 500.)
                    start_with_zero = self.configuration["pitch_ref"]["zero_start"])

                step_phi = tasks.Const(value=0.0)

            elif self.configuration["aspiration"] == "eval q":
                self.theta_trim = 0.21  # np.rad2deg(self.theta)
                step_theta = smoothed_step_sequence(
                    smooth_width=self.configuration["pitch_ref_elq"]["p_sw"],
                    times=self.configuration["pitch_ref_elq"]["p_sp_t"],
                    amplitudes=self.configuration["pitch_ref_elq"]["p_sp"])
                step_phi = smoothed_step_sequence(
                    smooth_width=self.configuration["roll_ref_elq"]["r_sw"],
                    times=self.configuration["roll_ref_elq"]["r_sp_t"],
                    amplitudes=self.configuration["roll_ref_elq"]["r_sp"])

            else:
                if not self.configuration["near_stall"]:
                    print("Evaluation reference")
                    self.theta_trim = np.rad2deg(self.theta) - 0.0006
                    step_theta  = smoothed_step_sequence(
                        smooth_width    = self.configuration["pitch_ref_e"]["p_sw"],
                        times           = self.configuration["pitch_ref_e"]["p_sp_t"],
                        amplitudes      = self.configuration["pitch_ref_e"]["p_sp"])
                    step_phi    = smoothed_step_sequence(
                        smooth_width    = self.configuration["roll_ref_e"]["r_sw"],
                        times           = self.configuration["roll_ref_e"]["r_sp_t"],
                        amplitudes      = self.configuration["roll_ref_e"]["r_sp"])

                else:
                    print("Evaluation near-stall reference")
                    self.theta_trim = np.rad2deg(self.theta) - 0.0006
                    step_theta  = smoothed_step_sequence(
                        smooth_width    = self.configuration["pitch_ref_stall"]["p_sw"],
                        times           = self.configuration["pitch_ref_stall"]["p_sp_t"],
                        amplitudes      = self.configuration["pitch_ref_stall"]["p_sp"])
                    step_phi    = tasks.Const(value=0.0)

                    #step_phi = smoothed_step_sequence(
                    #    smooth_width    = self.configuration["pitch_ref_stall"]["p_sw"],
                    #    times           = self.configuration["pitch_ref_stall"]["p_sp_t"],
                    #    amplitudes      = self.configuration["pitch_ref_stall"]["p_sp"])
                    #step_theta = tasks.Const(value=0.0)

            """
            else:
                if not self.eval_mode:
                    Warning(
                        'User reference signals have been given why env is not in evaluation mode.')
                step_theta  = kwargs['user_refs']['theta_ref']
                step_phi    = kwargs['user_refs']['phi_ref']
            """
            # add trim value for theta
            step_theta += tasks.Const(value = self.theta_trim) # 0., self.t_max, self.theta_trim)

            self.ref = [step_theta, step_phi, step_beta]

    def calc_reference_value(self):
        self.ref_values = np.asarray([np.deg2rad(ref_signal(self.t)) for ref_signal in self.ref])

    def get_controlled_state(self) -> float:
        """ Returns the value of the controlled states."""
        _crtl = np.asarray([self.theta, self.phi, self.beta])
        return _crtl[:self.n_actions]

    def calc_error(self) -> np.array:
        """ Updates the current error. """
        self.calc_reference_value()
        control_state               = self.get_controlled_state()
        self.error[:self.n_actions] = self.ref_values - control_state

    def get_reward(self) -> float:
        """ Returns the experience reward."""
        self.calc_error()
        reward_vec = np.abs(np.clip(self.error_scaler * self.error, -self.max_bound, self.max_bound))
        experiment_reward = - reward_vec.sum() / self.error.shape[0]
        return experiment_reward

    def get_cost(self) -> float:
        """ Returns the (binary) cost of the last transition."""
        if np.rad2deg(np.abs(self.alpha)) > 11.0 or np.rad2deg(np.abs(self.phi)) > 0.75 * self.max_phi or \
           self.V < self.V0/3:
            return 1
        return 0

    def incremental_control(self, action: np.ndarray) -> np.ndarray:
        """ Return low-pass filtered incremental control input to be given to the Citation model. """
        return self.last_u + action * self.dt

    def pad_action(self, action: np.ndarray) -> np.ndarray:
        """ Pad the control input with 0 to correspond to the Simulink dimensions. """
        citation_input = np.pad(action, (0, self.n_actions_full - self.n_actions), 'constant', constant_values = (0.))
        return citation_input

    def check_bounds(self) -> bool:

        if self.t >= self.t_max or np.abs(self.theta) > self.max_theta or np.abs(self.phi) > self.max_phi or self.H < 50:
            if self.configuration["aspiration"] == "train" or self.configuration["aspiration"] == "train_theta":
                if self.t >= self.t_max:
                    print('Finished at condition 1: Time limit :)')
            if np.abs(self.theta) > self.max_theta:
                print('Failed at condition 2: Theta :(')
            if np.abs(self.phi) > self.max_phi:
                print('Failed at condition 3: Phi :(')
            if self.H < 50:
                print('Failed at condition 4: Crashed on the ground :(')

            # Negative reward for crossing the bound and failing too soon (proportional to time)
            penalty = -1/self.dt * (self.t_max - self.t) * self.configuration["training"]["terrain_penalty"]
            self.t_fin  = self.t
            return True, penalty

        return False, 0.

    def get_observation(self):
        error       = self.error.flatten() #* np.array([2., 1., 4.]) #np.array([2., 1., 4.])
        states      = self.x[self.obs_idx]
        obs         = np.hstack((error, states))
        return obs

    def get_t_fin(self):
        return self.t_fin

    def reset(self, **kwargs) -> np.ndarray:
        # Reset time
        self.t      = 0.
        self.t_fin  = 0.

        self.track_list     = []
        self.states_list    = []
        self.action_list    = []
        self.error_list     = []
        self.reference_list = []
        self.time_list      = []
        self.p_list         = []
        self.reward_list    = []

        # Initalize the simulink model
        self.citation.initialize()

        # Make a full-zero input step to retreive the state
        self.last_u = np.zeros(self.n_actions)

        # Init state vector and observation vector
        _input = self.pad_action(self.last_u)
        self.x = self.citation.step(_input)

        # Init aicraft reference conditions
        self.V0 = self.V

        # Randomize reference signal sequence
        self.init_ref(**kwargs)

        # Build observation
        self.obs = self.get_observation()
        self.last_obs = self.obs[:]

        if self.use_incremental:
            self.obs = np.hstack((self.obs, self.last_u))

        return self.obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """ Step function which simualtes one step in time of the agent in the environment.

        Args:
            action (np.ndarray): Un-sclaled action in the interval [-1,1] to be taken.

        Returns:
            Tuple[np.ndarray, float, bool, dict]: new_state, obtained reward, is_done mask,
                                     info:{refference signal value, time, cost of step}
        """
        self.last_obs = self.obs

        #if self.t == 0:
        #    action = np.array([17/16-1, 19/17-1, 0.])

        # scale action to actuator rate bounds
        action = self.scale_action(action)   # scaled to actuator limits


        # incremental control input:
        if self.use_incremental:
            u = self.incremental_control(action)
        else:
            u = action

        # Citation step
        _input = self.pad_action(u)
        self.x = self.citation.step(_input)

        if self.gust:

            if (self.t >= 4) and (self.t <= 7):
                self.x[4] += d2r(2.5)

            #if (self.t >= 10) and (self.t <= 13):
            #    self.x[4] += d2r(2.5)

            #if (self.t >= 15) and (self.t <= 18):
            #    self.x[4] += d2r(2.5)

            #if (self.t >= 20) and (self.t <= 23):
            #    self.x[4] += d2r(2.5)

            #if (self.t >= 30) and (self.t <= 33):
            #    self.x[4] += d2r(2.5)

            #if (self.t >= 45) and (self.t <= 48):
            #    self.x[4] += d2r(2.5)

            #if (self.t >= 20) and (self.t <= 23):
            #    self.x[4] += d2r(2.5)

        # Reward using clipped error
        reward = self.get_reward()
        self.reward_list.append(reward)

        # Calculate cost:
        cost = self.get_cost()

        # Update observation based on perfect observations & actuator state
        self.obs    = self.get_observation()
        self.last_u = u
        if self.use_incremental:
            self.obs = np.hstack((self.obs, self.last_u))

        # Check for bounds and ad corresponding penalty for dying early
        is_done, penalty    = self.check_bounds()
        reward              += penalty

        # Step time:
        if not is_done:
            self.t += self.dt
            self.time_list.append(self.t)

        # Info:
        info = {
            "ref":  self.ref_values,
            "x":    self.x,
            "t":    self.t,
            "cost": cost,
        }

        self.track_list.append(self.x)
        self.reference_list.append(self.ref_values)

        return self.obs, reward, is_done, info

    def finish(self):
        """ Terminate the Simulink."""
        self.citation.terminate()



