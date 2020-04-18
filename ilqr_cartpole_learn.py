import argparse
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from future.builtins import input

from cartpole import default_params
from cartpole import CartpoleDraw
from plant import gTrig_np

from ilqr_cartpole import CartpoleILQRAgent
from ilqr_cartpole import CartpoleLQRAgent

def main(args):
    print("iLQR")
    print("====")
    x0 = np.array([0, 0, 0, 0]) + 0.1 * np.random.uniform(size=(4,))
    learner_params = default_params()
    plant_params = learner_params['params']['plant']
    plant_params['x0'] = x0
    plant = learner_params['plant_class'](**plant_params)
    scale_dims = np.array([4, 100, 500, 2 * np.pi])
    Q, R, Qf = iLQR_params(args.horizon, scale_dims=scale_dims)
    Qstab = np.diag([2, 2, 8, 20])
    Rstab = np.array([[300]])
    target = np.array([0, 0, 0, np.pi])
    agent_swingup = CartpoleILQRAgent(plant, Q, R, Qf, alpha=0, thresh=args.thresh, grad_clip=args.grad_clip)
    agent_stable = CartpoleLQRAgent(plant, Qstab, Rstab, target, np.array([[0.]]))
    agent_stable.get_controls()
    _, _, us = agent_swingup.get_controls(x0, target, args.horizon, max_iterations=args.iterations)
    xs, _ = agent_swingup.forward(x0, us)

    input('Press ENTER to simulate...')

    draw_cp = CartpoleDraw(plant)
    draw_cp.start()

    for i in range(3):
        plant.reset_state()
        apply_controller(plant, agent_swingup, agent_stable, learner_params['params'], args.horizon + 10, horizon=args.horizon)

def apply_controller(plant, agent_swingup, agent_stable, params, H, horizon=None):
    x_t, t = plant.get_plant_state()
    for i in range(H):
        x_t_ = gTrig_np(x_t[None, :], params['angle_dims']).flatten()
        if (horizon is None) or i < horizon:
            u_t = agent_swingup.policy(x_t, i)
        else:
            if (horizon is not None) and (i == horizon):
                print("Reached end of control sequence")
                print("Final state : {}".format(x_t))
                input("Press ENTER to continue...")
            u_t = agent_stable.policy(x_t, i)
        plant.apply_control(u_t)
        plant.step()
        x_t, t = plant.get_plant_state()
        if plant.done:
            break
    plant.stop()

def iLQR_params(horizon=None, scale_dims=None):
    if scale_dims is None:
        scale_dims = np.array([1,1,1,1])
#    Q = np.diag([1, 0, 0, 0])
#    R = np.array([[1]])
#    Q = np.diag([1, 1, 1, 1])
#    Qf = np.diag([5, 50, 10, 700])
    Q = np.diag([1e-2, 3e-3, 3e-1, 1e-1] / scale_dims)
    Qf = np.diag([2e-1, 1e-1, 1, 3000] / scale_dims)
    R = np.array([[5.0e-6]])
    if horizon is None:
        return Q, R, Qf
    return np.sqrt(Q / horizon), np.sqrt(R / horizon), np.sqrt(Qf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Swing up a cartpole with iLQR')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=5000)
    parser.add_argument('--thresh', type=float, default=1)
    parser.add_argument('--grad_clip', type=float, default=20)
    args = parser.parse_args()
    main(args)
