import pandas as pd

from h2000_v90 import citation as citation_h2000_v90
import numpy as np
import tasks
from matplotlib import pyplot as plt
import matplotlib as mpl

def pad_action(action: np.ndarray):
    """
    Pad action with 0 to correspond to the Simulink dimensions.
    """
    citation_input = np.pad(action, (0, 7), 'constant', constant_values = (0.))
    return citation_input


citation    = citation_h2000_v90
step_e      = tasks.Const(value = np.deg2rad(-5), t_start = 1.0, t_end = 3.0)
step_a      = tasks.Const(value = np.deg2rad(10), t_start = 4.0, t_end = 5.0)
step_r      = tasks.Const(value = 0.0)
states      = []
step_e_list = []
step_a_list = []
step_r_list = []

t_test = 1.05
print(t_test)
print('Test step e', step_e(t_test))
print('Test done')

times       = np.arange(0., 10.01, 0.01)

citation.initialize()
for t in times:

    actions = np.array([step_e(t), step_a(t), step_r(t)])
    step_e_list.append(step_e(t))
    step_a_list.append(step_a(t))
    step_r_list.append(step_r(t))

    inp     = pad_action(actions)
    x       = citation.step(inp)
    states.append(x)

states = np.array(states)
print('size', states.shape)

theta_list          = states[:, 7]
phi_list            = states[:, 6]
q_state_list        = states[:, 1]
p_state_list        = states[:, 0]
r_state_list        = states[:, 2]
alpha_state_list    = states[:, 4]
beta_list           = states[:, 5]
V_list              = states[:, 3]
height_list         = states[:, 9]

lw      = 1.5
dark    = 0.17

font = {'size': 15}
mpl.rc('font', **font)
mpl.rc('xtick', labelsize = 12)
mpl.rc('ytick', labelsize = 12)

bcg = '#F6F9FD'
bcg = '#F7F9FC'

fig, axis = plt.subplots(4, 3)
fig.subplots_adjust(hspace=0.1, wspace=.22, right=0.95, left=0.05)#, top = .99, bottom=0.06, right = 0.99, left = 0.05

axis[0, 0].plot(times, np.rad2deg(theta_list), linewidth = 1.2*lw, color = "#006C84", marker = "o", markevery = 100)
axis[0, 0].plot(times, np.rad2deg(theta_list), linewidth = 0.9*lw, color = "#F34A4A", marker = "D", markevery = (50,100))
axis[0, 0].grid(linestyle = 'dashed', alpha = 0.5)
axis[0, 0].set_ylabel(r"$\theta$ [deg]")
axis[0, 0].set_xlim((0., 10.))
axis[0, 0].set_facecolor(color = bcg)
axis[0, 0].tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)

axis[0, 1].plot(times, np.rad2deg(phi_list), linewidth = 1.5, color = "#006C84", marker = "o", markevery = 100)
axis[0, 1].plot(times, np.rad2deg(phi_list), linewidth = 1.5, color = "#F34A4A", marker = "D", markevery = (50,100))
axis[0, 1].grid(linestyle = 'dashed', alpha = 0.5)
axis[0, 1].set_ylabel(r"$\phi$ [deg]")
axis[0, 1].set_xlim((0., 10.))
axis[0, 1].set_facecolor(color = bcg)
axis[0, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axis[3, 0].plot(times, np.rad2deg(step_e_list), linewidth = 1.1*lw, color = "#04946c")
axis[3, 0].set_ylabel(r"$\delta_e$ [deg]")
axis[3, 0].set_xlim((0., 10.))
axis[3, 0].set_xlabel("Time [s]")
axis[3, 0].set_facecolor(color = bcg)
axis[3, 0].grid(linestyle = 'dashed', alpha = 0.5)

axis[3, 1].plot(times, np.rad2deg(step_a_list), linewidth = 1.1*lw, color = "#04946c")
axis[3, 1].set_ylabel(r"$\delta_a$ [deg]")
axis[3, 1].set_xlim((0., 10.))
axis[3, 1].set_xlabel("Time [s]")
axis[3, 1].set_facecolor(color = bcg)
axis[3, 1].grid(linestyle = 'dashed', alpha = 0.5)

axis[3, 2].plot(times, np.rad2deg(step_r_list), linewidth = 1.1*lw, color = "#04946c")
axis[3, 2].set_ylabel(r"$\delta_r$ [deg]")
axis[3, 2].set_xlim((0., 10.))
axis[3, 2].set_xlabel("Time [s]")
axis[3, 2].set_facecolor(color = bcg)
axis[3, 2].grid(linestyle = 'dashed', alpha = 0.5)

axis[1, 0].plot(times, np.rad2deg(q_state_list), linewidth = lw, color = "#006C84", marker = "o", markevery = 100)
axis[1, 0].plot(times, np.rad2deg(q_state_list), linewidth = lw, color = "#F34A4A", marker = "D", markevery = (50,100))
axis[1, 0].set_ylabel("q [deg/s]")
axis[1, 0].set_xlim((0., 10.))
axis[1, 0].grid(linestyle = 'dashed', alpha = 0.5)
axis[1, 0].set_facecolor(color = bcg)
axis[1, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axis[1, 1].plot(times, np.rad2deg(p_state_list), linewidth = lw, color = "#006C84", marker = "o", markevery = 100)
axis[1, 1].plot(times, np.rad2deg(p_state_list), linewidth = lw, color = "#F34A4A", marker = "D", markevery = (50,100))
axis[1, 1].set_ylabel("p [deg/s]")
axis[1, 1].set_xlim((0., 10.))
axis[1, 1].grid(linestyle = 'dashed', alpha = 0.5)
axis[1, 1].set_facecolor(color = bcg)
axis[1, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axis[1, 2].plot(times, np.rad2deg(r_state_list), linewidth = lw, color = "#006C84", marker = "o", markevery = 100)
axis[1, 2].plot(times, np.rad2deg(r_state_list), linewidth = lw, color = "#F34A4A", marker = "D", markevery = (50,100))
axis[1, 2].set_ylabel("r [deg/s]")
axis[1, 2].set_xlim((0., 10.))
axis[1, 2].grid(linestyle = 'dashed', alpha = 0.5)
axis[1, 2].set_facecolor(color = bcg)
axis[1, 2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axis[2, 0].plot(times, np.rad2deg(alpha_state_list), linewidth = lw, color = "#006C84", marker = "o", markevery = 100)
axis[2, 0].plot(times, np.rad2deg(alpha_state_list), linewidth = lw, color = "#F34A4A", marker = "D", markevery = (50,100))
axis[2, 0].set_ylabel(r"$\alpha$ [deg]")
axis[2, 0].set_xlim((0., 10.))
axis[2, 0].grid(linestyle = 'dashed', alpha = 0.5)
axis[2, 0].set_facecolor(color = bcg)
axis[2, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axis[0, 2].plot(times, np.rad2deg(beta_list), color = "#006C84", marker = "o", markevery = 100)
axis[0, 2].plot(times, np.rad2deg(beta_list), color = "#F34A4A", marker = "D", markevery = (50,100))
axis[0, 2].set_ylabel(r"$\beta$ [deg]")
axis[0, 2].set_xlim((0., 10.))
axis[0, 2].grid(linestyle = 'dashed', alpha = 0.5)
axis[0, 2].set_facecolor(color = bcg)
axis[0, 2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axis[2, 1].plot(times, V_list, linewidth = lw, color = "#006C84", marker = "o", markevery = 100)
axis[2, 1].plot(times, V_list, linewidth = lw, color = "#F34A4A", marker = "D", markevery = (50,100))
axis[2, 1].set_ylabel("V [m/s]")
axis[2, 1].set_xlim((0., 10.))
axis[2, 1].grid(linestyle = 'dashed', alpha = 0.5)
axis[2, 1].set_facecolor(color = bcg)
axis[2, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axis[2, 2].plot(times, height_list, linewidth = lw, color = "#006C84", marker = "o", markevery = 100)
axis[2, 2].plot(times, height_list, linewidth = lw, color = "#F34A4A", marker = "D", markevery = (50,100))
axis[2, 2].set_ylabel("h [m]")
axis[2, 2].set_xlim((0., 10.))
axis[2, 2].grid(linestyle = 'dashed', alpha = 0.5)
axis[2, 2].set_facecolor(color = bcg)
axis[2, 2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


'''
np.set_printoptions(threshold = np.inf)
print('Theta [deg]:', np.rad2deg(theta_list))
print('Phi [deg]:', np.rad2deg(phi_list))
print('Elevator input [deg]', np.rad2deg(step_e_list))
print('Aileron input [deg]', np.rad2deg(step_a_list))
print('Rudder input [deg]', np.rad2deg(step_r_list))
print('q [deg/s]:', np.rad2deg(q_state_list))
print('p [deg/s]:', np.rad2deg(p_state_list))
print('r [deg/s]:', np.rad2deg(r_state_list))
print('Angle of attack [deg]:', np.rad2deg(alpha_state_list))
print('Angle of sideslip [deg]:', np.rad2deg(beta_list))
print('Airspeed [m/s]:', V_list)
print('Altitude [m]:', height_list)
'''

# Validation of seed
file    = ('../../data/dsac-Cessna500-nonlin-attitude/Original/dsac_Cessna500_nonlin_attitude_2023_09_09_00_26_28_0000--s-2110/progress.csv')
data    = pd.read_csv(file, header = 0)
steps   = data["Epoch"]
returns = data["evaluation/Average Returns"]
lw2     = 2.

plt.figure("DSAC seed validation")
axplot = plt.axes()
axplot.set_facecolor(color = bcg)
plt.plot(steps, returns, linewidth = lw2, color = "#006C84", marker = "o", markevery = 2, markersize = 4.77,
         label = "DSAC training run 1 (seed = 2110)")
plt.plot(steps, returns, linewidth = 0.8*lw2, color = "#F34A4A", marker = "D", markevery = (1, 2), markersize = 4.77,
         linestyle = (0, (5, 5)), label = "DSAC training run 2 (seed = 2110)")
plt.grid()
plt.legend(loc = 4)
plt.xlabel(r'Episode [-]')
plt.ylabel(r'Average return [-]')
plt.show()


