import numpy as np
import matplotlib.pyplot as plt
from lqr import AnalyticILQR
from lqr import LQRStabilizer

class CartpoleLQRAgent(LQRStabilizer):
    def __init__(self, plant, Q, R, x, u):
        super(CartpoleLQRAgent, self).__init__(Q, R, x, u)
        self._plant = plant
        self._plant_params = self._extract_plant_params()

    def dynamics(self, x_, u):
        # x = [x xdot tdot t]
        dt = 0.8 * self._plant.dt

        x = x_.copy()

        l, m, M, b, g = self._plant_params

        xdot = x[1]
        tdot = x[2]
        sin_t = np.sin(x[3])
        cos_t = np.cos(x[3])

        xddot_num = 2*m*l*tdot*tdot*sin_t + 3*m*g*sin_t*cos_t + 4*(u - b*xdot)
        xddot_den = 4 * (M + m) - 3 * m * cos_t * cos_t
        xddot = xddot_num / xddot_den

        tddot_num = -3*(m*l*tdot*tdot*sin_t*cos_t + 2*((M+m)*g*sin_t + (u - b*xdot)*cos_t))
        tddot_den = l * (4 * (M + m) - 3 * m * cos_t * cos_t)
        tddot = tddot_num / tddot_den

        x[0] += xdot * dt
        x[1] += xddot * dt
        x[2] += tddot * dt
        x[3] += tdot * dt

        return x

    def _extract_plant_params(self):
        params = self._plant.params
        l = params['l']
        m = params['m']
        M = params['M']
        b = params['b']
        g = params['g']

        return l, m, M, b, g

class CartpoleILQRAgent(AnalyticILQR):
    def __init__(self, plant, Q, R, Qf=None, alpha=0, thresh=1e-2, grad_clip=20):
        super(CartpoleILQRAgent, self).__init__(Q, R, Qf=Qf, alpha=alpha)
        self._plant = plant
        self._plant_params = self._extract_plant_params()
        self._thresh = thresh
        self._grad_clip = grad_clip
        self._Ks = None
        self._ds = None
        self._xs = None
        self._us = None

    def dynamics(self, x_, u):
        # x = [x xdot tdot t]
        dt = self._plant.dt

        x = x_.copy()

        l, m, M, b, g = self._plant_params

        xdot = x[1]
        tdot = x[2]
        sin_t = np.sin(x[3])
        cos_t = np.cos(x[3])

        xddot_num = 2*m*l*tdot*tdot*sin_t + 3*m*g*sin_t*cos_t + 4*(u - b*xdot)
        xddot_den = 4 * (M + m) - 3 * m * cos_t * cos_t
        xddot = xddot_num / xddot_den

        tddot_num = -3*(m*l*tdot*tdot*sin_t*cos_t + 2*((M+m)*g*sin_t + (u - b*xdot)*cos_t))
        tddot_den = l * (4 * (M + m) - 3 * m * cos_t * cos_t)
        tddot = tddot_num / tddot_den

        x[0] += xdot * dt
        x[1] += xddot * dt
        x[2] += tddot * dt
        x[3] += tdot * dt

        return x

    def get_controls(self, x0, target, horizon, max_iterations=100, verbose=True):
#        us = np.array([[1 * np.sin(0.1 * i)] for i in range(horizon-1)])
        us = np.array([[np.random.normal()] for i in range(horizon-1)])
        loss = 0
        for iteration in range(1, max_iterations+1):
            xs, loss_ = self.forward(self.preprocess(x0), us, target)
            loss = loss_.copy()
            Ks, ds = self.backward(xs, us, target)
            x = self.preprocess(x0)
            dus = np.zeros_like(us)
            for i in range(len(us)):
                du = Ks[i].dot(x - xs[i]) + ds[i]
                if abs(du.item()) - self._grad_clip > 0:
                    du = self._grad_clip * du / abs(du)
                us[i] += du
                dus[i] = du
                x = self.preprocess(self.dynamics(x, us[i]))
                xs[i] = x.copy()
            normdu = np.sqrt(dus.T.dot(dus).item())
            if normdu < self._thresh:
                if verbose:
                    print('iLQR converged.')
                break
            if verbose:
                delta = xs[-1] - target
                normdelta = np.sqrt(delta.T.dot(delta).item())
                fmtstr = "Episode {:>4}/{} :: Loss: {:>10.4f} | norm(du): {:>8.2f} | delta: {:>12} | norm(delta): {:>8.2f}"
                print(fmtstr.format(iteration, max_iterations, loss.item() / horizon, normdu, str(np.abs(np.around(delta, 2))), normdelta))
        self._Ks = Ks
        self._ds = ds
        self._xs = xs
        self._us = us
        return Ks, ds, us

    def policy(self, x_, i):
        x = self.preprocess(x_)
        if i >= len(self._Ks):
            i = len(self._Ks) - 1
        return self._Ks[i].dot(x - self._xs[i]) + self._ds[i] + self._us[i]

    def preprocess(self, x_):
        x = x_.copy()
        if x[3] < 0:
            x[3] += 2 * np.pi
        elif x[3] > 2 * np.pi:
            x[3] -= 2 * np.pi
        return x

    def _extract_plant_params(self):
        params = self._plant.params
        l = params['l']
        m = params['m']
        M = params['M']
        b = params['b']
        g = params['g']

        return l, m, M, b, g
