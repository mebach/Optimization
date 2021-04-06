import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
import VTOLParam as P
from signalGenerator import signalGenerator
from VTOLAnimation import VTOLAnimation
from dataPlotter import dataPlotter
from VTOLDynamics import VTOLDynamics
from VTOL_MPC import VTOL_MPC


# instantiate VTOL, controller, and reference classes
VTOL = VTOLDynamics()
controller = VTOL_MPC()
z_reference = signalGenerator(amplitude=2.5, frequency=0.01, y_offset=3.0)
h_reference = signalGenerator(amplitude=3.0, frequency=0.01, y_offset=5.0)
disturbance = signalGenerator(amplitude=0.1)
# f_l = signalGenerator(amplitude=50.0, frequency=0.5)
# f_r = signalGenerator(amplitude=50.0, frequency=0.5)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = VTOLAnimation()

t = P.t_start  # times starts at t_start
y = VTOL.h()

while t < P.t_end:

    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot

    # updates control and dynamics at faster simulation rate
    while t < t_next_plot:
        h_ref = h_reference.square(t)
        z_ref = z_reference.square(t)
        r = np.array([[z_ref], [h_ref]])
        dfl = disturbance.step(t)
        dfr = disturbance.step(t)
        d = np.array([[dfl], [dfr]])
        n = 0.0  # noise.random(t)
        x = VTOL.state
        u = controller.update(r, x)
        y = VTOL.update(u)
        t = t + P.Ts

    animation.update(VTOL.state, 1.0)
    z_history, h_history, theta_history = dataPlot.update(t, VTOL.state, z_ref, h_ref, u.item(0), u.item(1))

    # the pause causes the figure to be displayed during the simulation
    plt.pause(0.0001)

t = P.t_start
state = np.array([[np.array(z_history), np.array(h_history), np.array(theta_history)*np.pi/180]])
i = 0
while t < P.t_end:
    # print('entered while loop')
    # print(state[:,:,i])
    t_next_plot = t + P.t_plot
    animation.update(state[:, :, i], 1.0)
    i += 1
    plt.pause(0.0001)
# keeps the program from closing until the user presses a button
print('Press key to close')
plt.waitforbuttonpress()
plt.close()