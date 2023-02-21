# build neural networks for regression
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torchlaplace import laplace_reconstruct
import time
import logging
from config import default_config
from copy import deepcopy
from overlay import setup_logger, create_env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger()


def cartpole_dynamics_dt_delay(state, perturbed_action, ts, delay, ACTION_LOW=-3.0, ACTION_HIGH=3.0, nu=1, friction=False, friction_cart= 5e-4, friction_pole = 2e-6):
    perturbed_action = perturbed_action[:, -(delay + 1), :nu]
    assert state.shape[0] == perturbed_action.shape[0]
    assert perturbed_action.shape[1] == 1
    if len(ts.shape) >= 2:
        assert ts.shape[1] == 1
    ts = ts.view(-1, 1)
    if state.shape[-1] == 5:
        x = state[:, 0].view(-1, 1)
        x_dot = state[:, 1].view(-1, 1)
        costheta = state[:, 2].view(-1, 1)
        sintheta = state[:, 3].view(-1, 1)
        theta_dot = state[:, 4].view(-1, 1)
        C = (costheta**2 + sintheta**2).detach()
        costheta, sintheta = costheta/C, sintheta/C
        theta = torch.atan2(sintheta/C, costheta/C)
    else:
        x = state[:, 0].view(-1, 1)
        x_dot = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)
        theta_dot = state[:, 3].view(-1, 1)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

    gravity = 9.8
    force_mag = 3.0
    masscart = 1.0
    masspole = 0.1
    length = 1.0  # actually half the pole's length
    total_mass = (masspole + masscart)
    polemass_length = (masspole * length)

    u = perturbed_action
    u = torch.clamp(u, ACTION_LOW, ACTION_HIGH)

    force = u*force_mag
    if friction:
        temp = (force + polemass_length * theta_dot * theta_dot * sintheta - friction_cart * torch.sign(x_dot)) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp - friction_pole * theta_dot / polemass_length) / \
            (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
    else:
        temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / \
            (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    if state.shape[-1] == 5:
        new_theta_dot = theta_dot + thetaacc * ts
        new_theta = theta + theta_dot * ts
        new_costheta = torch.cos(new_theta)
        new_sintheta = torch.sin(new_theta)
        new_x_dot = x_dot + xacc * ts
        new_x = x + x_dot * ts
        state = torch.cat((new_x, new_x_dot, new_costheta, new_sintheta, new_theta_dot), dim=1)
    else:
        new_theta_dot = theta_dot + thetaacc * ts
        new_theta = theta + theta_dot * ts
        new_x_dot = x_dot + xacc * ts
        new_x = x + x_dot * ts
        state = torch.cat((new_x, new_x_dot, new_theta, new_theta_dot), dim=1)
    return state

def acrobot_dynamics_dt_delay(state, perturbed_action, ts, delay, ACTION_LOW=-5.0, ACTION_HIGH=5.0, nu=2, friction=False):
    perturbed_action = perturbed_action[:, -(delay + 1), :nu]
    assert state.shape[0] == perturbed_action.shape[0]
    assert perturbed_action.shape[1] == 2
    if len(ts.shape) >= 2:
        assert ts.shape[1] == 1
    ts = ts.view(-1, 1)
    if state.shape[-1] == 6:
        costtheta1 = state[:, 0].view(-1, 1)
        sintheta1 = state[:, 1].view(-1, 1)
        costtheta2 = state[:, 2].view(-1, 1)
        sintheta2 = state[:, 3].view(-1, 1)
        dtheta1 = state[:, 4].view(-1, 1)
        dtheta2 = state[:, 5].view(-1, 1)
        C1 = (costtheta1**2 + sintheta1**2).detach()
        costheta1, sintheta1 = costtheta1/C1, sintheta1/C1
        theta1 = torch.atan2(sintheta1/C1, costheta1/C1)
        C2 = (costtheta2**2 + sintheta2**2).detach()
        costheta2, sintheta2 = costtheta2/C2, sintheta2/C2
        theta2 = torch.atan2(sintheta2/C2, costheta2/C2)
    elif state.shape[-1] == 4:
        theta1 = state[:, 0].view(-1, 1)
        theta2 = state[:, 1].view(-1, 1)
        dtheta1 = state[:, 2].view(-1, 1)
        dtheta2 = state[:, 3].view(-1, 1)

    m1 = 1.  #: [kg] mass of link 1
    m2 = 1.  #: [kg] mass of link 2
    l1 = 1.  # [m]
    lc1 = 0.5  #: [m] position of the center of mass of link 1
    lc2 = 0.5  #: [m] position of the center of mass of link 2
    I1 = 1.  #: moments of inertia for both links
    I2 = 1.  #: moments of inertia for both links
    g = 9.8

    u = perturbed_action
    u = torch.clamp(u, ACTION_LOW, ACTION_HIGH)
    d1 = m1 * lc1 ** 2 + m2 * \
        (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * torch.cos(theta2)) + I1 + I2

    d2 = m2 * (lc2 ** 2 + l1 * lc2 * torch.cos(theta2)) + I2

    phi2 = m2 * lc2 * g * torch.cos(theta1 + theta2 - np.pi / 2.)
    phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * torch.sin(theta2) \
        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * torch.sin(theta2)  \
        + (m1 * lc1 + m2 * l1) * g * torch.cos(theta1 - np.pi / 2) + phi2

    ddtheta2 = (u[:, 0].view(-1, 1) + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * torch.sin(theta2) - phi2) \
        / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)

    ddtheta1 = -(u[:, 1].view(-1, 1) + d2 * ddtheta2 + phi1) / d1

    new_dtheta1 = dtheta1 + ddtheta1 * ts
    new_dtheta2 = dtheta2 + ddtheta2 * ts
    new_theta1 = theta1 + dtheta1 * ts
    new_theta2 = theta2 + dtheta2 * ts

    if state.shape[-1] == 4:
        return torch.cat((new_theta1, new_theta2, new_dtheta1, new_dtheta2), dim=1)
    elif state.shape[-1] == 6:
        return torch.cat((torch.cos(new_theta1), torch.sin(new_theta1), torch.cos(new_theta2), torch.sin(new_theta2), new_dtheta1, new_dtheta2), dim=1)


def pendulum_dynamics_dt_delay(state, perturbed_action, ts, delay, ACTION_LOW=-2.0, ACTION_HIGH=2.0, nu=1, friction=False):
    perturbed_action = perturbed_action[:, -(delay + 1), :nu]
    assert state.shape[0] == perturbed_action.shape[0]
    assert perturbed_action.shape[1] == 1
    if len(ts.shape) >= 2:
        assert ts.shape[1] == 1
    ts = ts.view(-1, 1)
    if state.shape[-1] == 2:
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)
    elif state.shape[-1] == 3:
        costh = state[:, 0].view(-1, 1)
        sinth = state[:, 1].view(-1, 1)
        thdot = state[:, 2].view(-1, 1)
        C = (costh**2 + sinth**2).detach()
        costheta, sintheta = costh/C, sinth/C
        th = torch.atan2(sintheta/C, costheta/C)

    g = 10
    m = 1
    l = 1

    u = perturbed_action
    u = torch.clamp(u, ACTION_LOW, ACTION_HIGH)

    if state.shape[-1] == 2:
        newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + torch.pi) + 3. / (m * l ** 2) * u) * ts
        newth = th + thdot * ts
        state = torch.cat((newth, newthdot), dim=1)
        return state
    elif state.shape[-1] == 3:
        newth = th + thdot * ts
        new_costheta = torch.cos(newth)
        new_sintheta = torch.sin(newth)
        newthdot = thdot + (-3*g/(2*l) * torch.sin(th+np.pi) + 3./(m*l**2)*u) * ts
        state = torch.cat((new_costheta, new_sintheta, newthdot), dim=1)
        return state

def cartpole_dynamics_dt_latent_reduced(state, prev_state, perturbed_action, ts, ACTION_LOW=-3.0, ACTION_HIGH=3.0):
    perturbed_action = perturbed_action.view(perturbed_action.shape[0], perturbed_action.shape[-1])
    # perturbed_action = torch.zeros_like(perturbed_action)
    assert state.shape[0] == perturbed_action.shape[0]
    assert perturbed_action.shape[1] == 1
    if len(ts.shape) >= 2:
        assert ts.shape[1] == 1
    ts = ts.view(-1, 1)
    if state.shape[-1] == 5:
        x = state[:, 0].view(-1, 1)
        xp = prev_state[:, 0].view(-1, 1)
        x_dot = (x - xp) / ts
        # x_dot = state[:, 1].view(-1, 1)
        costheta = state[:, 1].view(-1, 1)
        sintheta = state[:, 2].view(-1, 1)
        # theta_dot = state[:, 4].view(-1, 1)
        C = (costheta**2 + sintheta**2).detach()
        costheta, sintheta = costheta/C, sintheta/C
        theta = torch.atan2(sintheta/C, costheta/C)

        costhetap = prev_state[:, 1].view(-1, 1)
        sinthetap = prev_state[:, 2].view(-1, 1)
        # theta_dot = state[:, 4].view(-1, 1)
        C = (costhetap**2 + sinthetap**2).detach()
        costhetap, sinthetap = costhetap/C, sinthetap/C
        thetap = torch.atan2(sinthetap/C, costhetap/C)

        theta_dot = (theta - thetap) / ts
        # print(f'theta_dot: {theta_dot}')
    else:
        x = state[:, 0].view(-1, 1)
        xp = prev_state[:, 0].view(-1, 1)
        x_dot = (x - xp) / ts
        # x_dot = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)
        thetap = prev_state[:, 2].view(-1, 1)
        theta_dot = (theta - thetap) / ts
        # theta_dot = state[:, 3].view(-1, 1)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

    gravity = 9.8
    force_mag = 3.0
    masscart = 1.0
    masspole = 0.1
    length = 1.0  # actually half the pole's length
    total_mass = (masspole + masscart)
    polemass_length = (masspole * length)

    u = perturbed_action
    u = torch.clamp(u, ACTION_LOW, ACTION_HIGH)

    force = u*force_mag
    temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / \
        (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    new_theta_dot = theta_dot + thetaacc * ts
    # new_theta = theta + theta_dot * ts
    new_theta = theta + new_theta_dot * ts
    new_costheta = torch.cos(new_theta)
    new_sintheta = torch.sin(new_theta)
    new_x_dot = x_dot + xacc * ts
    # new_x = x + x_dot * ts
    new_x = x + new_x_dot * ts
    state = torch.cat((new_x, new_costheta, new_sintheta), dim=1)
    return state


def cartpole_dynamics_dt_latent(state, prev_state, perturbed_action, ts, ACTION_LOW=-3.0, ACTION_HIGH=3.0):
    perturbed_action = perturbed_action.view(perturbed_action.shape[0], perturbed_action.shape[-1])
    # perturbed_action = torch.zeros_like(perturbed_action)
    assert state.shape[0] == perturbed_action.shape[0]
    assert perturbed_action.shape[1] == 1
    if len(ts.shape) >= 2:
        assert ts.shape[1] == 1
    ts = ts.view(-1, 1)
    if state.shape[-1] == 5:
        x = state[:, 0].view(-1, 1)
        xp = prev_state[:, 0].view(-1, 1)
        x_dot = (x - xp) / ts
        # x_dot = state[:, 1].view(-1, 1)
        costheta = state[:, 2].view(-1, 1)
        sintheta = state[:, 3].view(-1, 1)
        # theta_dot = state[:, 4].view(-1, 1)
        C = (costheta**2 + sintheta**2).detach()
        costheta, sintheta = costheta/C, sintheta/C
        theta = torch.atan2(sintheta/C, costheta/C)

        costhetap = prev_state[:, 2].view(-1, 1)
        sinthetap = prev_state[:, 3].view(-1, 1)
        # theta_dot = state[:, 4].view(-1, 1)
        C = (costhetap**2 + sinthetap**2).detach()
        costhetap, sinthetap = costhetap/C, sinthetap/C
        thetap = torch.atan2(sinthetap/C, costhetap/C)

        theta_dot = (theta - thetap) / ts
        # print(f'theta_dot: {theta_dot}')
    else:
        x = state[:, 0].view(-1, 1)
        xp = prev_state[:, 0].view(-1, 1)
        x_dot = (x - xp) / ts
        # x_dot = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)
        thetap = prev_state[:, 2].view(-1, 1)
        theta_dot = (theta - thetap) / ts
        # theta_dot = state[:, 3].view(-1, 1)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

    gravity = 9.8
    force_mag = 3.0
    masscart = 1.0
    masspole = 0.1
    length = 1.0  # actually half the pole's length
    total_mass = (masspole + masscart)
    polemass_length = (masspole * length)

    u = perturbed_action
    u = torch.clamp(u, ACTION_LOW, ACTION_HIGH)

    force = u*force_mag
    temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / \
        (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    if state.shape[-1] == 5:
        new_theta_dot = theta_dot + thetaacc * ts
        # new_theta = theta + theta_dot * ts
        new_theta = theta + new_theta_dot * ts
        new_costheta = torch.cos(new_theta)
        new_sintheta = torch.sin(new_theta)
        new_x_dot = x_dot + xacc * ts
        # new_x = x + x_dot * ts
        new_x = x + new_x_dot * ts
        state = torch.cat((new_x, new_x_dot, new_costheta, new_sintheta, new_theta_dot), dim=1)
    else:
        new_theta_dot = theta_dot + thetaacc * ts
        # new_theta = theta + theta_dot * ts
        new_theta = theta + new_theta_dot * ts
        new_x_dot = x_dot + xacc * ts
        # new_x = x + x_dot * ts
        new_x = x + new_x_dot * ts
        state = torch.cat((new_x, new_x_dot, new_theta, new_theta_dot), dim=1)
    return state


def cartpole_dynamics_dt(state, perturbed_action, ts, ACTION_LOW=-3.0, ACTION_HIGH=3.0):
    assert state.shape[0] == perturbed_action.shape[0]
    assert perturbed_action.shape[1] == 1
    if len(ts.shape) >= 2:
        assert ts.shape[1] == 1
    ts = ts.view(-1, 1)
    if state.shape[-1] == 5:
        x = state[:, 0].view(-1, 1)
        x_dot = state[:, 1].view(-1, 1)
        costheta = state[:, 2].view(-1, 1)
        sintheta = state[:, 3].view(-1, 1)
        theta_dot = state[:, 4].view(-1, 1)
        C = (costheta**2 + sintheta**2).detach()
        costheta, sintheta = costheta/C, sintheta/C
        theta = torch.atan2(sintheta/C, costheta/C)
    else:
        x = state[:, 0].view(-1, 1)
        x_dot = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)
        theta_dot = state[:, 3].view(-1, 1)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

    gravity = 9.8
    force_mag = 3.0
    masscart = 1.0
    masspole = 0.1
    length = 1.0  # actually half the pole's length
    total_mass = (masspole + masscart)
    polemass_length = (masspole * length)

    u = perturbed_action
    u = torch.clamp(u, ACTION_LOW, ACTION_HIGH)

    force = u*force_mag
    temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / \
        (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    if state.shape[-1] == 5:
        new_theta_dot = theta_dot + thetaacc * ts
        new_theta = theta + theta_dot * ts
        new_costheta = torch.cos(new_theta)
        new_sintheta = torch.sin(new_theta)
        new_x_dot = x_dot + xacc * ts
        new_x = x + x_dot * ts
        state = torch.cat((new_x, new_x_dot, new_costheta, new_sintheta, new_theta_dot), dim=1)
    else:
        new_theta_dot = theta_dot + thetaacc * ts
        new_theta = theta + theta_dot * ts
        new_x_dot = x_dot + xacc * ts
        new_x = x + x_dot * ts
        state = torch.cat((new_x, new_x_dot, new_theta, new_theta_dot), dim=1)
    return state


def pendulum_dynamics_dt(state, perturbed_action, ts, ACTION_LOW=-2.0, ACTION_HIGH=2.0):
    assert state.shape[0] == perturbed_action.shape[0]
    assert perturbed_action.shape[1] == 1
    if len(ts.shape) >= 2:
        assert ts.shape[1] == 1
    ts = ts.view(-1, 1)
    if state.shape[-1] == 2:
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)
    elif state.shape[-1] == 3:
        costh = state[:, 0].view(-1, 1)
        sinth = state[:, 1].view(-1, 1)
        thdot = state[:, 2].view(-1, 1)
        C = (costh**2 + sinth**2).detach()
        costheta, sintheta = costh/C, sinth/C
        th = torch.atan2(sintheta/C, costheta/C)

    g = 10
    m = 1
    l = 1

    u = perturbed_action
    u = torch.clamp(u, ACTION_LOW, ACTION_HIGH)

    if state.shape[-1] == 2:
        newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + torch.pi) + 3. / (m * l ** 2) * u) * ts
        newth = th + thdot * ts
        state = torch.cat((newth, newthdot), dim=1)
        return state
    elif state.shape[-1] == 3:
        newth = th + thdot * ts
        new_costheta = torch.cos(newth)
        new_sintheta = torch.sin(newth)
        newthdot = thdot + (-3*g/(2*l) * torch.sin(th+np.pi) + 3./(m*l**2)*u) * ts
        state = torch.cat((new_costheta, new_sintheta, newthdot), dim=1)
        return state


def acrobot_dynamics_dt(state, perturbed_action, ts, ACTION_LOW=-5.0, ACTION_HIGH=5.0):
    assert state.shape[0] == perturbed_action.shape[0]
    assert perturbed_action.shape[1] == 2
    if len(ts.shape) >= 2:
        assert ts.shape[1] == 1
    ts = ts.view(-1, 1)
    if state.shape[-1] == 6:
        costtheta1 = state[:, 0].view(-1, 1)
        sintheta1 = state[:, 1].view(-1, 1)
        costtheta2 = state[:, 2].view(-1, 1)
        sintheta2 = state[:, 3].view(-1, 1)
        dtheta1 = state[:, 4].view(-1, 1)
        dtheta2 = state[:, 5].view(-1, 1)
        C1 = (costtheta1**2 + sintheta1**2).detach()
        costheta1, sintheta1 = costtheta1/C1, sintheta1/C1
        theta1 = torch.atan2(sintheta1/C1, costheta1/C1)
        C2 = (costtheta2**2 + sintheta2**2).detach()
        costheta2, sintheta2 = costtheta2/C2, sintheta2/C2
        theta2 = torch.atan2(sintheta2/C2, costheta2/C2)
    elif state.shape[-1] == 4:
        theta1 = state[:, 0].view(-1, 1)
        theta2 = state[:, 1].view(-1, 1)
        dtheta1 = state[:, 2].view(-1, 1)
        dtheta2 = state[:, 3].view(-1, 1)

    m1 = 1.  #: [kg] mass of link 1
    m2 = 1.  #: [kg] mass of link 2
    l1 = 1.  # [m]
    lc1 = 0.5  #: [m] position of the center of mass of link 1
    lc2 = 0.5  #: [m] position of the center of mass of link 2
    I1 = 1.  #: moments of inertia for both links
    I2 = 1.  #: moments of inertia for both links
    g = 9.8

    u = perturbed_action
    u = torch.clamp(u, ACTION_LOW, ACTION_HIGH)
    d1 = m1 * lc1 ** 2 + m2 * \
        (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * torch.cos(theta2)) + I1 + I2

    d2 = m2 * (lc2 ** 2 + l1 * lc2 * torch.cos(theta2)) + I2

    phi2 = m2 * lc2 * g * torch.cos(theta1 + theta2 - np.pi / 2.)
    phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * torch.sin(theta2) \
        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * torch.sin(theta2)  \
        + (m1 * lc1 + m2 * l1) * g * torch.cos(theta1 - np.pi / 2) + phi2

    ddtheta2 = (u[:, 0].view(-1, 1) + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * torch.sin(theta2) - phi2) \
        / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)

    ddtheta1 = -(u[:, 1].view(-1, 1) + d2 * ddtheta2 + phi1) / d1

    new_dtheta1 = dtheta1 + ddtheta1 * ts
    new_dtheta2 = dtheta2 + ddtheta2 * ts
    new_theta1 = theta1 + dtheta1 * ts
    new_theta2 = theta2 + dtheta2 * ts

    if state.shape[-1] == 4:
        state = torch.cat((new_theta1, new_theta2, new_dtheta1, new_dtheta2), dim=1)
    elif state.shape[-1] == 6:
        state = torch.cat((torch.cos(new_theta1), torch.sin(new_theta1), torch.cos(new_theta2), torch.sin(new_theta2), new_dtheta1, new_dtheta2), dim=1)
    return state
