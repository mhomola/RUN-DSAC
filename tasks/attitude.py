"""Module that defines the attitude tracking tasks."""
from abc import ABC

import numpy as np

from tasks.base_task import BaseTask

def cos_step(t: float, t0: float, w: float):
    """Smooth cosine step function."""
    a = -(np.cos(1 / w * np.pi * (t - t0)) - 1) / 2
    if t0 >= 0:
        a = 0.0 if t < t0 else a
    a = 1.0 if t > t0 + w else a
    return a

class Attitude(BaseTask, ABC):
    """Task to track a reference signal of the attitude."""

    tracked_states = ["theta", "phi", "beta"]

    def __init__(self, env):
        super().__init__(env)

    @property
    def scale(self):
        """The scale of each state tracked in the task."""
        return 1 / np.deg2rad([30, 30, 7.5])


class AttitudeTrain(Attitude):
    """Task to train the attitude controller."""

    def __init__(self, env):
        super().__init__(env)
        self.theta_max = np.deg2rad(20)  # maximum pitch angle [deg]
        self.phi_max = np.deg2rad(20)  # maximum roll angle [deg]

        increment = [1, -1, 0]

        self.theta_increment = [1, 0, -1, 1]
        self.phi_increment = [0, 1, 0, -1]
        self.theta_amplitude = np.deg2rad(np.random.choice([5, 10, 15]))
        self.phi_amplitude = np.deg2rad(np.random.choice([5, 10, 15]))

    def __str__(self):
        return "att_train"

    def reference(self) -> np.ndarray:
        """Reference signal."""
        t = self.env.current_time

        # Theta reference
        theta_amp = self.theta_amplitude
        theta_increment = self.theta_increment
        theta_ref = theta_amp * np.sum(
            [
                theta_increment[i] * cos_step(t, 5 * i, 5)
                for i in range(len(self.theta_increment))
            ]
        )
        theta_ref = np.clip(theta_ref, -self.theta_max, self.theta_max)

        # Phi reference
        phi_amp = self.phi_amplitude
        phi_increment = self.phi_increment
        phi_ref = phi_amp * np.sum(
            [
                phi_increment[i] * cos_step(t, 5 * i, 5)
                for i in range(len(self.phi_increment))
            ]
        )
        phi_ref = np.clip(phi_ref, -self.phi_max, self.phi_max)

        beta_ref = 0

        return np.hstack((theta_ref, phi_ref, beta_ref))


class AttitudeEval(AttitudeTrain):
    """Task to evaluate the attitude controller."""

    def __init__(self, env):
        super().__init__(env)
        self.theta_amplitude = np.deg2rad(10)
        self.phi_amplitude = np.deg2rad(5)
        self.theta_increment = [1, 0, 1, 0, -1, 0, -1]
        self.phi_increment = [1, 0, -1, 0, -1, 0, 1]

    def __str__(self):
        return "att_eval"


class SineAttitude(Attitude):
    """Task to track a sinusoidal reference signal of the citation atittude."""

    def __init__(self, env):
        super().__init__(env)
        self.amp_theta = np.deg2rad(
            np.random.choice([20, 10, -10, -20], 1)
        )  # amplitude [rad]
        self.amp_phi = np.deg2rad(
            np.random.choice([40, 20, -20, -40], 1)
        )  # amplitude [rad]

    def __str__(self):
        return "sin_att"

    def reference(self) -> np.ndarray:
        """Reference signal."""
        t = self.env.current_time
        period = 6  # s

        # Theta reference
        amp_theta = self.amp_theta  # amplitude [rad]
        theta_ref = amp_theta * np.sin(2 * np.pi / period * t)

        # Phi reference
        amp_phi = self.amp_phi  # amplitude [rad]
        phi_ref = amp_phi * np.sin(2 * np.pi / period * t)

        # Beta reference
        beta_ref = 0

        return np.hstack((theta_ref, phi_ref, beta_ref))


class SinAttitudeEvaluate(SineAttitude):
    def __init__(self, env):
        super().__init__(env)
        self.amp_phi = np.deg2rad(20)  # amplitude [rad]
        self.amp_theta = np.deg2rad(10)  # amplitude [rad]

    def __str__(self):
        return "sin_att_eval"