from abc import ABCMeta, abstractmethod

import gym
import numpy as np
import torch
from gym import spaces
from gym.utils import seeding
from torchdiffeq import odeint

from envs.oderl.utils.utils import numpy_to_torch


class BaseEnv(gym.Env, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        dt,
        n,
        m,
        act_rng,
        obs_trans,
        name,
        state_actions_names,
        device,
        solver,
        obs_noise,
        ts_grid,
        ac_rew_const=0.01,
        vel_rew_const=0.01,
        n_steps=200,
    ):
        self.dt = dt
        self.n = n
        self.m = m
        self.act_rng = act_rng
        self.obs_trans = obs_trans
        self.name = name
        self.reward_range = [-ac_rew_const * act_rng**2, 1.0]  # pyright: ignore
        self.state_actions_names = state_actions_names
        self.ac_rew_const = ac_rew_const
        self.vel_rew_const = vel_rew_const
        self.obs_noise = obs_noise
        self.ts_grid = ts_grid
        # derived
        self.viewer = None
        self.action_space = spaces.Box(low=-self.act_rng, high=self.act_rng, shape=(self.m,))
        self.seed()
        self.ac_lb = numpy_to_torch(self.action_space.low, device=device)  # pyright: ignore
        self.ac_ub = numpy_to_torch(self.action_space.high, device=device)  # pyright: ignore
        self.set_solver(method=solver)

        self.n_steps = n_steps
        self.time_step = 0

    def set_solver(self, method="euler", rtol=1e-6, atol=1e-9, num_bins=None):
        if num_bins is None:
            if method == "euler":
                num_bins = 1
            elif method == "rk4":
                num_bins = 50
            else:
                num_bins = 1
        self.solver = {
            "method": method,
            "rtol": rtol,
            "atol": atol,
            "step_size": self.dt / num_bins,
        }

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def device(self):
        return self.ac_lb.device

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_obs(self):
        if self.obs_trans:
            torch_state = torch.tensor(self.state).unsqueeze(0)  # pyright: ignore  # pylint: disable=no-member
            # return list(self.torch_transform_states(torch_state)[0].numpy())
            return self.torch_transform_states(torch_state)[0].numpy()
        else:
            return self.state  # pyright: ignore  # pylint: disable=no-member

    def reward(self, obs, a):
        return self.np_obs_reward_fn(obs) + self.np_ac_reward_fn(a)  # pyright: ignore  # pylint: disable=no-member

    def diff_reward(self, s, a):
        if not isinstance(s, torch.Tensor) or not isinstance(a, torch.Tensor):
            raise NotImplementedError("Differentiable reward only accepts torch.Tensor inputs\n")
        return self.diff_obs_reward_(s) + self.diff_ac_reward_(a)

    def build_time_grid(self, T=None, only_one_step=True, device=None):
        if device is None:
            device = self.device
        if only_one_step:
            if self.ts_grid == "fixed":
                ts = torch.arange(2, device=device) * self.dt
            elif self.ts_grid == "uniform" or self.ts_grid == "random":
                ts = torch.cat(
                    (
                        torch.tensor([0.0], device=device),
                        (torch.rand(1, device=device) * 2 * self.dt),
                    )
                )
            elif self.ts_grid == "exp":
                ts = torch.cat(
                    (
                        torch.tensor([0.0], device=device),
                        torch.distributions.exponential.Exponential(1 / self.dt)
                        .sample([1])  # pyright: ignore
                        .to(device),
                    )
                )
            else:
                raise ValueError("Time grid parameter is wrong!")
            return ts
        else:
            if self.ts_grid == "fixed":
                ts = torch.arange(T, device=device) * self.dt  # pyright: ignore
            elif self.ts_grid == "uniform" or self.ts_grid == "random":
                ts = (torch.rand(T, device=device) * 2 * self.dt).cumsum(0)  # pyright: ignore
            elif self.ts_grid == "exp":
                ts = torch.distributions.exponential.Exponential(1 / self.dt).sample([T])  # pyright: ignore
                ts = ts.cumsum(0).to(device)
            else:
                raise ValueError("Time grid parameter is wrong!")
            return ts

    def integrate_system(self, T, g, s0=None, N=1, return_states=False):
        """Returns torch tensors
        states  - [N,T,n] where s0=[N,n]
        actions - [N,T,m]
        rewards - [N,T]
        ts      - [N,T]
        """
        with torch.no_grad():
            s0 = (
                torch.stack([numpy_to_torch(self.reset()) for _ in range(N)]).to(self.device)
                if s0 is None
                else numpy_to_torch(s0)
            )
            s0 = self.obs2state(s0)
            ts = self.build_time_grid(T)

            def odefnc(t, s):
                a = g(self.torch_transform_states(s), t)  # 1,m
                return self.torch_rhs(s, a)

            st = odeint(
                odefnc,
                s0,
                ts,
                rtol=self.solver["rtol"],
                atol=self.solver["atol"],
                method=self.solver["method"],
            )
            at = torch.stack([g(self.torch_transform_states(s_), t_) for s_, t_ in zip(st, ts)])
            rt = self.diff_reward(st, at)  # T,N
            if len(rt.shape) > 1:
                st, at, rt = st.permute(1, 0, 2), at.permute(1, 0, 2), rt.T  # pyright: ignore
            st_obs = self.torch_transform_states(st)
            st_obs += torch.randn_like(st_obs) * self.obs_noise  # pyright: ignore
            returns = [st_obs, at, rt, torch.stack([ts] * st_obs.shape[0])]
            if return_states:
                returns.append(st)
            return returns

    def batch_integrate_system_double_time(self, is0s, actions, device=None):
        """Returns torch tensors
        states  - [N,T,n] where s0=[N,n]
        actions - [N,T,m]
        rewards - [N,T]
        ts      - [N,T]
        """
        # from tqdm import tqdm
        if device is None:
            device = self.device
        # from time import time
        # t0 = time()
        with torch.no_grad():
            # print(f'{time() - t0}')
            s0s = self.obs2state(is0s)
            ts = self.build_time_grid(device=device, only_one_step=False, T=3)
            sb_l = []
            sn_l = []
            # for a in tqdm(actions):
            for a in actions:
                ab = a.view(1, -1).repeat(s0s.shape[0], 1)

                def odefnc(t, s):
                    return self.torch_rhs(s, ab)  # pylint: disable=cell-var-from-loop

                st = odeint(
                    odefnc,
                    s0s,
                    ts,
                    rtol=self.solver["rtol"],
                    atol=self.solver["atol"],
                    method=self.solver["method"],
                )
                sb_l.append(st[-2, :, :])  # pyright: ignore
                sn_l.append(st[-1, :, :])  # pyright: ignore
            sn = torch.stack(sn_l)
            sb = torch.stack(sb_l)
            # print(f'OUT {time() - t0}')
            sn = sn.view(-1, sn.shape[2])
            sb = sb.view(-1, sb.shape[2])
            s0s = self.torch_transform_states(s0s)
            sb = self.torch_transform_states(sb)
            # print(f'{time() - t0}')
            sn = self.torch_transform_states(sn)
            if len(actions.shape) == 1:
                a0 = actions.repeat_interleave(s0s.shape[0]).view(-1, 1)
            else:
                a0 = actions.repeat_interleave(s0s.shape[0]).view(-1, actions.shape[1])
            # a0 = actions.view(1,-1).repeat(s0s.shape[0],1).view(-1,1)
            # print(f'{time() - t0}')
            s0s_out = s0s.unsqueeze(0).repeat(actions.shape[0], 1, 1).view(-1, s0s.shape[1])
            if self.obs_noise != 0.0:
                sn += torch.randn_like(sn) * self.obs_noise
            # print(f' FIN {time() - t0}')
            return s0s_out, a0, sb, sn, ts[1]

    def batch_integrate_system(self, is0s, actions, device=None):
        """Returns torch tensors
        states  - [N,T,n] where s0=[N,n]
        actions - [N,T,m]
        rewards - [N,T]
        ts      - [N,T]
        """
        # from tqdm import tqdm
        if device is None:
            device = self.device
        # from time import time
        # t0 = time()
        with torch.no_grad():
            # print(f'{time() - t0}')
            s0s = self.obs2state(is0s)
            ts = self.build_time_grid(device=device)
            sn_l = []
            # for a in tqdm(actions):
            for a in actions:
                ab = a.view(1, -1).repeat(s0s.shape[0], 1)

                def odefnc(t, s):
                    return self.torch_rhs(s, ab)  # pylint: disable=cell-var-from-loop

                st = odeint(
                    odefnc,
                    s0s,
                    ts,
                    rtol=self.solver["rtol"],
                    atol=self.solver["atol"],
                    method=self.solver["method"],
                )
                sn_l.append(st[-1, :, :])  # pyright: ignore
            sn = torch.stack(sn_l)
            # print(f'OUT {time() - t0}')
            sn = sn.view(-1, sn.shape[2])
            s0s = self.torch_transform_states(s0s)
            # print(f'{time() - t0}')
            sn = self.torch_transform_states(sn)
            if len(actions.shape) == 1:
                a0 = actions.repeat_interleave(s0s.shape[0]).view(-1, 1)
            else:
                a0 = actions.repeat_interleave(s0s.shape[0]).view(-1, actions.shape[1])
            # a0 = actions.view(1,-1).repeat(s0s.shape[0],1).view(-1,1)
            # print(f'{time() - t0}')
            s0s_out = s0s.unsqueeze(0).repeat(actions.shape[0], 1, 1).view(-1, s0s.shape[1])
            if self.obs_noise != 0.0:
                sn += torch.randn_like(sn) * self.obs_noise
            # print(f' FIN {time() - t0}')
            return s0s_out, a0, sn, ts[-1]

    def torch_transform_states(self, state):
        if self.obs_trans:
            raise NotImplementedError
        else:
            return state

    def obs2state(self, state):
        if self.obs_trans:
            raise NotImplementedError
        else:
            return state

    def np_terminating_reward(self, state):  # [...,n]
        return np.zeros(state.shape[:-1]) * 0.0

    def trigonometric2angle(self, costheta, sintheta):
        C = (costheta**2 + sintheta**2).detach()
        costheta, sintheta = costheta / C, sintheta / C
        theta = torch.atan2(sintheta / C, costheta / C)
        return theta

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def torch_rhs(self, state, action):
        raise NotImplementedError

    @abstractmethod
    def diff_obs_reward_(self, s):
        raise NotImplementedError

    @abstractmethod
    def diff_ac_reward_(self, a):
        raise NotImplementedError

    @abstractmethod
    def render(self, mode, **kwargs):  # pylint: disable=signature-differs
        raise NotImplementedError
