import numpy as np
import matplotlib.pyplot as plt
from tasks.random_signals import randomized_cosine_step_sequence
import yaml
import matplotlib as mpl

mpl.use('TkAgg', force=True)

configuration_file = "../configs/Attitude/config.yaml"
with open(configuration_file, 'r', encoding="utf-8") as f:
    configuration = yaml.load(f, Loader=yaml.FullLoader)

step_theta = randomized_cosine_step_sequence(
                    t_max           = 30,
                    ampl_max        = configuration["pitch_ref"]["amplitude"],
                    block_width     = configuration["pitch_ref"]["block_w"],
                    smooth_width    = configuration["pitch_ref"]["smooth_w"],
                    n_levels        = configuration["pitch_ref"]["n_levels"],
                    vary_timings    = configuration["pitch_ref"]["var_w"],
                    start_with_zero = configuration["pitch_ref"]["zero_start"])

step_phi = randomized_cosine_step_sequence(
                    t_max           = 30,
                    ampl_max        = configuration["roll_ref"]["amplitude"],
                    block_width     = configuration["roll_ref"]["block_w"],
                    smooth_width    = configuration["roll_ref"]["smooth_w"],
                    n_levels        = configuration["roll_ref"]["n_levels"],
                    vary_timings    = configuration["roll_ref"]["var_w"],
                    start_with_zero = configuration["roll_ref"]["zero_start"])

time = np.arange(0, 30, .01)

ref_theta   = []
ref_phi     = []

for t in time:
    ref_theta.append(step_theta(t))
    ref_phi.append(step_phi(t))

font = {'size': 16.5}
mpl.rc('font', **font)
mpl.rc('xtick', labelsize = 13)
mpl.rc('ytick', labelsize = 13)
lw      = 1.5
dark    = 0.22
bcg     = '#F9FBFE'


fig, axs = plt.subplots(2)

axs[0].plot(time,   ref_theta,    linewidth = lw,     color = "#006C84")
axs[1].plot(time,   ref_phi,      linewidth = lw,     color = "#006C84")

axs[0].grid(linestyle = 'dashed', alpha = 0.5)
axs[0].set_ylabel(r"$\theta_{ref}$ [deg]")
axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axs[1].grid(linestyle = 'dashed', alpha = 0.5)
axs[1].set_ylabel(r"$\phi_{ref}$ [deg]")
axs[1].set_xlabel(r"Time [s]")

axs[0].set_facecolor(color = bcg)
axs[1].set_facecolor(color = bcg)
plt.show()


