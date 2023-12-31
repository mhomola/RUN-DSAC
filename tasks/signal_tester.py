import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg',force=True)
import numpy as np

font = {'size': 15}
mpl.rc('font', **font)

from tasks import randomized_step_sequence, randomized_cosine_step_sequence
from tools.utils import d2r, r2d

signal_def      = randomized_step_sequence(t_max = 30, ampl_max = 5, block_width = 3, n_levels = 7)
signal_def2     = randomized_cosine_step_sequence(t_max = 30, ampl_max = 25, block_width = 8.0, smooth_width = 3.0,
                                                  vary_timings = 0.1, n_levels = 15)
#signal_def      = three_two_one_one(t_start=2.0, ampl=5.0, block_width=3.0)
t               = np.linspace(0, 30, 1000)
signal          = []
signal2         = []

for time in t:
    signal.append(signal_def(time))
    signal2.append(signal_def2(time))

plt.figure("Test - Randomized Cosine Step Sequence")
plt.plot(t, signal2, linewidth = 1.75, color = '#006C84')
plt.xlabel("Time [s]")
plt.ylabel(r"$\alpha_{ref}$ [deg]")
#plt.ylim(-6, 6)
plt.grid()
plt.show()
