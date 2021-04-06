import numpy as np

class VTOLController:
    def __init__(self):


    def update(self, r, y):
        z = y.item(0)
        h = y.item(1)
        theta = y.item(2)

        z_r = r.item(0)
        h_r = r.item(1)

        F_tilde = self.hCtrl.PID(h_r, h, flag=False)
        F = F_tilde + P.Fe

        theta_r = self.zCtrl.PID(z_r, z, flag=False)
        tau = self.thetaCtrl.PID(theta_r, theta, flag=False)


        return np.array([[F], [tau]])








