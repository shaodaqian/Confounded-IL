# This is an example for confounded imitation learning where we consider an environment of drones with wind and weather as hidden confounders

import os

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box

from models.utils import device


# x: state, the position of the drone
# a: action, the control input of the drone
# u1: weather, the weather condition. observable by the expert but not the learner
# u1 \in (0,1)
# u2: wind, the wind speed and direction. not observable by the expert and the learner

# The goal is to patrol between -1 and 1

class NoisyDroneEnv:
    def __init__(self, ue_k=3, uo_k=3, x0=(3, 1), uo_low=0.3, uo_high=1., ue_mu=0.5, ue_sigma=1., action_noise=0.2,
                 patrol_range=5.):
        self.ue_k = ue_k
        self.uo_k = uo_k
        self.x0 = x0
        self.uo_low = uo_low
        self.uo_high = uo_high
        self.ue_mu = ue_mu
        self.ue_sigma = ue_sigma
        self.action_noise = action_noise
        self.patrol_range = patrol_range
        self.positive_flag = True

        # np.random.seed(666)

    def transition(self, x, a, uo, ue, uo_next, ue_next):
        # new_x=x+(uo*a)+ue_next#+u2+np.random.normal(0.3,0.1)
        if isinstance(x, float):
            clean_x = np.sign(x) * 1.0 * uo
            new_x = clean_x + ue_next
        else:
            clean_x = np.sign(x) * np.full(np.array(x).shape, 1.0) * uo
            new_x = clean_x + ue_next
        return new_x, clean_x

    def expert_policy(self, x, uo, ue):
        a_1 = np.clip(-x / uo, -1, 1)
        return a_1

    # noise_level=0.2 # effect of wind as noise on action
    def noised_policy(self, x, uo, ue):
        return self.expert_policy(x, uo, ue) + self.action_noise * ue

    def reset_env(self):
        self.cur_x = np.random.normal(self.x0[0], self.x0[1])
        self.uo_queue = np.random.uniform(self.uo_low, self.uo_high, self.uo_k)
        self.ue_queue = np.random.normal(self.ue_mu, np.sqrt(self.ue_k) * self.ue_sigma, self.ue_k)

        # self.u2=None

    def step(self, a=None, policy=None):
        # pass in policy if it requires knowledge of hidden confounders
        old_x = self.cur_x

        uo = np.average(self.uo_queue)
        ue = np.average(self.ue_queue)
        self.uo_queue = np.append(self.uo_queue[1:], np.random.uniform(self.uo_low, self.uo_high, 1))
        self.ue_queue = np.append(self.ue_queue[1:], np.random.normal(self.ue_mu, np.sqrt(self.ue_k) * self.ue_sigma, 1))
        uo_next = np.average(self.uo_queue)
        ue_next = np.average(self.ue_queue)


        if a is None:
            assert policy is not None
            a = policy(old_x, uo, ue)
        new_x, clean_x = self.transition(old_x, a, uo, ue, uo_next, ue_next)

        self.cur_x = new_x
        r = 1 - clean_x

        return old_x, a, new_x, uo, ue, r

    def get_dataset(self, steps):
        self.reset_env()
        dataset = []
        for i in range(steps):
            x, a, new_x, uo, ue, r = self.step(policy=self.noised_policy)
            dataset.append([x, a, uo, ue])
            # print(f"Step {i}: x={x}, u1={u1}, u2={u2}, a={a}")
        return np.array(dataset)

    def evaluate(self, policy, steps=10000):
        self.reset_env()
        total_reward = 0
        total_error = 0
        for i in range(steps):
            x, a, new_x, uo, ue, r = self.step(policy=policy)
            total_reward += r
            total_error += (self.expert_policy(x, uo, ue) - a) ** 2

        return total_reward / steps, total_error / steps

    def history_evaluate(self, policy, policy_history=3, steps=10000):
        self.reset_env()
        history = [self.cur_x, 0] * policy_history
        total_reward = 0
        total_error = 0
        for i in range(steps):
            a_h = policy(torch.tensor(history).float().view(-1, 2 * policy_history).to(device)).item()
            x, a, new_x, uo, ue, r = self.step(a=a_h)
            total_reward += r
            total_error += (self.expert_policy(x, uo, ue) - a) ** 2
            history.pop(-1)
            history.pop(-1)
            history.insert(0, a)
            history.insert(0, new_x)
        return total_reward / steps, total_error / steps

    def dfiv_evaluate(self, policy, policy_history=3, steps=10000,
                      state_scaler=None, action_scaler=None):

        self.reset_env()
        history = [self.cur_x, 0] * policy_history

        total_reward = 0
        total_error = 0

        for _ in range(steps):
            a_h = policy(torch.tensor(history).float().view(-1, 2 * policy_history).to(device)).item()

            # if state_scaler:
            #     a_h = state_scaler.transform(np.expand_dims(a_h, 0))
            x, a, new_x, uo, ue, r = self.step(a=a_h)

            # if action_scaler:
            #     action = action_scaler.inverse_transform(action)

            total_reward += r
            total_error += (self.expert_policy(x, uo, ue) - a) ** 2
            history.pop(-1)
            history.pop(-1)
            history.insert(0, a)
            history.insert(0, new_x)
        return total_reward / steps, total_error / steps

class HCWrapper(gym.Wrapper):
    def __init__(self, env, uo_vis=False, ue_k=3, uo_k=10, uo_low=0.1, uo_high=3., ue_mu=0., ue_sigma=0.2,
                 action_noise=1., control_loss=0.1):
        super().__init__(env)
        # self.sticky = np.random.uniform(0.1, 2.0)
        self.ue_k = ue_k
        self.uo_k = uo_k
        self.uo_low = uo_low
        self.uo_high = uo_high
        self.ue_mu = ue_mu
        self.ue_sigma = ue_sigma
        self.action_noise = action_noise
        self.control_loss = control_loss

        self.env = env
        self.env.reset()
        self.uo_vis = uo_vis
        print(self.env.observation_space)  # 18 with pos 17 without
        self.dt = self.env.unwrapped.dt
        if self.uo_vis:
            self.observation_space = Box(low=-np.inf * np.ones(env.observation_space.shape[0] + 1),
                                         high=np.inf * np.ones(env.observation_space.shape[0] + 1),
                                         dtype=np.float32)

    def get_sticky(self):
        # return (self.uo_high-self.uo_low)*(np.sin(self.t/self.uo_k)+1.)/2+self.uo_low
        return np.clip(np.average(self.uo_queue), 0.1, 2)

    def reset(self, seed=666, **kwargs):
        self.uo_queue = np.random.uniform(self.uo_low, self.uo_high, self.uo_k)
        self.ue_queue = np.random.normal(self.ue_mu, np.sqrt(self.ue_k) * self.ue_sigma, self.ue_k)
        self.sticky = self.get_sticky()
        self.ue = np.average(self.ue_queue)
        obs, info = self.env.reset()
        if self.uo_vis:
            obs = np.concatenate([obs, [self.sticky]])
        # else:
        #     obs = np.concatenate([obs])
        return obs.astype(np.float32), info

    def confounded_step(self, action):
        cf_action = action + self.action_noise * self.ue
        next_obs, reward, done, truncated, info = self.step(action)
        next_obs[1:] += self.ue
        return next_obs, reward, cf_action, done, truncated, info

    def step(self, action):
        self.sticky = self.get_sticky()
        self.uo_queue = np.append(self.uo_queue[1:], np.random.uniform(self.uo_low, self.uo_high, 1))
        self.ue_queue = np.append(self.ue_queue[1:],
                                  np.random.normal(self.ue_mu, np.sqrt(self.ue_k) * self.ue_sigma, 1))
        self.ue = np.average(self.ue_queue)

        p = self.env.unwrapped.data.qpos[0]
        next_obs, _, done, truncated, info = self.env.step(action)
        p_prime = self.env.unwrapped.data.qpos[0]
        p_prime = p + (p_prime - p) / self.sticky
        # print('new p and stickness',p_prime,self.sticky)
        next_obs[0] = p_prime
        self.env.unwrapped.set_state(next_obs[:9], next_obs[9:])
        vel = (p_prime - p) / (self.dt)

        if self.uo_vis:
            next_obs = np.concatenate([next_obs, [self.sticky]])
        forward_reward = -1.0 * 1 * (vel - 1) ** 2
        ctrl_cost = self.control_loss * 1e-1 * np.sum(np.square(action))
        reward = forward_reward - ctrl_cost
        return next_obs.astype(np.float32), reward, done, truncated, info

    def evaluate_expert(self, expert, steps=500, episodes=100, confounded=False):
        total_reward = 0
        total_error = 0

        sum = 0
        for traj in range(episodes):
            obs, _ = self.reset()
            for i in range(steps):
                action, _state = expert.predict(obs, deterministic=True)
                sum += np.mean(np.square(action))
                if confounded:
                    cf_action = action + self.action_noise * self.ue
                    obs, r, done, truncated, info = self.step(cf_action)
                    obs += self.ue
                else:
                    obs, r, _, _, _ = self.step(action)
                    cf_action = action

                total_reward += r
                total_error += np.mean((action - cf_action) ** 2)
        print(sum / (steps * episodes), 'norm of expert')
        return total_reward / (steps * episodes), total_error / (steps * episodes)

    def evaluate(self, policy, expert, steps=500, episodes=100, confounded=False):
        policy.eval()
        total_reward = 0
        total_error = 0
        for traj in range(episodes):
            obs, _ = self.reset()

            # loc=obs[0]
            # obs[0]=-0.05*100
            for _ in range(steps):
                action = policy(torch.tensor(obs[:-1], dtype=torch.float32).to(device)).numpy(force=True)
                exp_act = expert.predict(obs, deterministic=True)[0]
                if confounded:
                    x, r, _, _, _, _ = self.confounded_step(action)
                else:
                    x, r, _, _, _ = self.step(action)
                obs = x
                total_reward += r
                total_error += np.mean((exp_act - action) ** 2)

        return total_reward / (steps * episodes), total_error / (steps * episodes)

    def dfiv_evaluate(self, policy, expert, policy_history=3, steps=500, episodes=100,
                      confounded=False, state_scaler=None, action_scaler=None):
        # policy.eval()
        total_reward = 0
        total_error = []
        for traj in range(episodes):
            obs, _ = self.reset()
            history = [torch.tensor(obs[:-1])] * policy_history

            for _ in range(steps):
                # print(torch.cat(history,dim=-1).shape)
                hist = torch.cat(history, dim=-1)
                hist[0] = (hist[0] - hist[18]) * 20
                hist = hist[:18]
                if state_scaler:
                    hist = state_scaler.transform(np.expand_dims(hist, 0))

                action = policy(torch.tensor(hist))
                if action_scaler:
                    action = action_scaler.inverse_transform(action)
                action = np.squeeze(action)
                exp_act = expert.predict(obs, deterministic=True)[0]
                # print(action,exp_act)
                if confounded:
                    x, r, _, _, _, _ = self.confounded_step(action)
                else:
                    x, r, _, _, _ = self.step(action)
                obs = x
                total_reward += r
                # total_error += np.mean((exp_act - action) ** 2)
                total_error.append((exp_act - action) ** 2)

                history.pop(-1)
                history.insert(0, torch.tensor(obs[:-1]))
        err = np.mean(np.array(total_error), 0)

        return total_reward / (steps * episodes), err

    def history_evaluate(self, policy, expert, policy_history=3, steps=500, episodes=100, use_action=False,
                         confounded=False):
        # policy.eval()
        total_reward = 0
        total_error = []
        for traj in range(episodes):
            obs, _ = self.reset()
            if use_action:
                history = [torch.tensor(obs[:-1]),
                           torch.zeros(self.env.action_space.shape, dtype=torch.float32)] * policy_history
            else:
                history = [torch.tensor(obs[:-1])] * policy_history

            for _ in range(steps):
                # print(torch.cat(history,dim=-1).shape)
                hist = torch.cat(history, dim=-1).to(device)
                hist[0] = (hist[0] - hist[18] - 0.0) * 20
                hist = hist[:18]
                action = policy(hist).cpu()
                exp_act = expert.predict(obs, deterministic=True)[0]
                if confounded:
                    x, r, _, _, _, _ = self.confounded_step(action.numpy(force=True))
                else:
                    x, r, _, _, _ = self.step(action.numpy(force=True))
                obs = x
                total_reward += r
                # total_error += np.mean((exp_act - action.numpy(force=True)) ** 2)
                total_error.append((exp_act - action.numpy(force=True)) ** 2)

                if use_action:
                    history.pop(-1)
                    history.insert(0, action)

                history.pop(-1)
                history.insert(0, torch.tensor(obs[:-1]))

        err = np.mean(np.array(total_error), 0)
        return total_reward / (steps * episodes), err

    def gen_expert_demos(self, dir_name, file_name, model, num_trajs, steps, save=True, confounded=False):
        trajs = dict()
        rewards = []
        for traj in range(num_trajs):
            total_reward = 0
            obs, _ = self.reset()
            done = False
            states = []
            actions = []
            step = 0
            while not done and step < steps:
                states.append(obs)  # masking context
                action, _state = model.predict(obs, deterministic=True)
                # print(action)
                if confounded:
                    obs, reward, action, done, truncated, info = self.confounded_step(action)
                else:
                    obs, reward, done, truncated, info = self.step(action)
                total_reward += reward
                actions.append(action)

                if done:
                    break
                step += 1

            trajs[str(traj)] = {'states': np.array(
                states), 'actions': np.array(actions)}
            rewards.append(total_reward / steps)
            # print('Finished traj:', traj, 'Average Reward:', total_reward / steps)
        print("Avg Reward:", np.mean(rewards))
        if save:
            np.savez(os.path.join('experts', dir_name, file_name), env=dir_name,
                     num_trajs=num_trajs,
                     mean_reward=np.mean(rewards),
                     std_reward=np.std(rewards),
                     **trajs)

        return np.mean(rewards)


class AntWrapper(gym.Wrapper):
    def __init__(self, env, uo_vis=False, ue_k=3, uo_k=10, uo_low=0.1, uo_high=3., ue_mu=0., ue_sigma=0.2,
                 action_noise=1., control_loss=0.1):
        super().__init__(env)
        # self.sticky = np.random.uniform(0.0, 1.5)

        self.ue_k = ue_k
        self.uo_k = uo_k
        self.uo_low = uo_low
        self.uo_high = uo_high
        self.ue_mu = ue_mu
        self.ue_sigma = ue_sigma
        self.action_noise = action_noise
        self.control_loss = control_loss

        self.env = env
        self.env.reset()
        self.uo_vis = uo_vis
        print(self.env.observation_space)  # 29 with pos 27 without
        self.dt = self.env.unwrapped.dt
        if self.uo_vis:
            self.observation_space = Box(low=-np.inf * np.ones(env.observation_space.shape[0] + 1),
                                         high=np.inf * np.ones(env.observation_space.shape[0] + 1),
                                         dtype=np.float32)

    def get_sticky(self):
        # return (self.uo_high-self.uo_low)*(np.sin(self.t/self.uo_k)+1.)/2+self.uo_low
        return np.clip(np.average(self.uo_queue), 0.1, 2)

    def reset(self, seed=666, **kwargs):
        self.uo_queue = np.random.uniform(self.uo_low, self.uo_high, self.uo_k)
        self.ue_queue = np.random.normal(self.ue_mu, np.sqrt(self.ue_k) * self.ue_sigma, self.ue_k)
        self.sticky = self.get_sticky()
        self.ue = np.average(self.ue_queue)
        obs, info = self.env.reset()
        if self.uo_vis:
            obs = np.concatenate([obs, [self.sticky]])
        # else:
        #     obs = np.concatenate([obs])
        return obs.astype(np.float32), info

    def confounded_step(self, action):
        cf_action = action + self.action_noise * self.ue
        next_obs, reward, done, truncated, info = self.step(action)
        next_obs[1:] += self.ue
        return next_obs, reward, cf_action, done, truncated, info

    def step(self, action):
        self.sticky = self.get_sticky()
        self.uo_queue = np.append(self.uo_queue[1:], np.random.uniform(self.uo_low, self.uo_high, 1))
        self.ue_queue = np.append(self.ue_queue[1:], np.random.normal(self.ue_mu, np.sqrt(self.ue_k) * self.ue_sigma, 1))
        self.ue = np.average(self.ue_queue)

        # p = self.env.env.parts['torso'].get_position()[:2]
        p = self.env.unwrapped.data.qpos[0]
        next_obs, _, done, truncated, info = self.env.step(action)
        # p_prime = self.env.env.parts['torso'].get_position()[:2]
        p_prime = self.env.unwrapped.data.qpos[0]

        p_prime = p + (p_prime - p) / self.sticky
        next_obs[0] = p_prime

        self.env.unwrapped.set_state(next_obs[:15], next_obs[15:])

        vel = (p_prime - p) / (self.dt)

        if self.uo_vis:
            next_obs = np.concatenate([next_obs, [self.sticky]])
        # else:
        #     next_obs = np.concatenate([next_obs])
        forward_reward = -1.0 * 1 * (vel - 1) ** 2
        ctrl_cost = self.control_loss * 1e-1 * np.sum(np.square(action))
        living_reward = 1  # for expert training
        reward = living_reward + forward_reward - ctrl_cost
        # reward = forward_reward - ctrl_cost

        return next_obs.astype(np.float32), reward, done, truncated, info

    def evaluate_expert(self, expert, steps=500, episodes=100, confounded=False):
        total_reward = 0
        total_error = 0

        for traj in range(episodes):
            obs, _ = self.reset()
            for i in range(steps):
                action, _state = expert.predict(obs, deterministic=True)
                if confounded:
                    cf_action = action + self.action_noise * self.ue
                    obs, r, done, truncated, info = self.step(cf_action)
                    obs += self.ue
                    # obs, r,action, _, _, _ = self.confounded_step(action)
                else:
                    obs, r, _, _, _ = self.step(action)
                    cf_action = action

                total_reward += r
                total_error += np.mean((action - cf_action) ** 2)

        return total_reward / (steps * episodes), total_error / (steps * episodes)

    def evaluate(self, policy, expert, steps=500, episodes=100, confounded=False):
        policy.eval()
        total_reward = 0
        total_error = 0
        for traj in range(episodes):
            obs, _ = self.reset()

            # loc=obs[0]
            # obs[0]=-0.05*100
            for _ in range(steps):
                action = policy(torch.tensor(obs[:-1], dtype=torch.float32).to(device)).numpy(force=True)
                exp_act = expert.predict(obs, deterministic=True)[0]

                if confounded:
                    x, r, _, _, _, _ = self.confounded_step(action)
                else:
                    x, r, _, _, _ = self.step(action)
                obs = x
                total_reward += r
                total_error += np.mean((exp_act - action) ** 2)

        return total_reward / (steps * episodes), total_error / (steps * episodes)

    def history_evaluate(self, policy, expert, policy_history=3, steps=500, episodes=100, use_action=False,
                         confounded=False):
        policy.eval()
        total_reward = 0
        total_error = 0
        for traj in range(episodes):
            obs, _ = self.reset()
            if use_action:
                history = [torch.tensor(obs[:-1]),
                           torch.zeros(self.env.action_space.shape, dtype=torch.float32)] * policy_history
            else:
                history = [torch.tensor(obs[:-1])] * policy_history

            for _ in range(steps):
                # print(torch.cat(history,dim=-1).shape)
                hist = torch.cat(history, dim=-1).to(device)
                hist[0] = (hist[0] - hist[29] - 0.0) * 20
                hist = hist[:29]
                action = policy(hist).cpu()
                exp_act = expert.predict(obs, deterministic=True)[0]
                if confounded:
                    x, r, _, _, _, _ = self.confounded_step(action.numpy(force=True))
                else:
                    x, r, _, _, _ = self.step(action.numpy(force=True))
                obs = x
                total_reward += r
                total_error += np.mean((exp_act - action.numpy(force=True)) ** 2)

                if use_action:
                    history.pop(-1)
                    history.insert(0, action)

                history.pop(-1)
                history.insert(0, torch.tensor(obs[:-1]))

        return total_reward / (steps * episodes), total_error / (steps * episodes)

    def gen_expert_demos(self, dir_name, file_name, model, num_trajs, steps, save=True, confounded=False):
        trajs = dict()
        rewards = []
        for traj in range(num_trajs):
            total_reward = 0
            obs, _ = self.reset()
            done = False
            states = []
            actions = []
            step = 0
            while not done and step < steps:
                states.append(obs)  # masking context
                action, _state = model.predict(obs, deterministic=True)
                # print(action)
                if confounded:
                    obs, reward, action, done, truncated, info = self.confounded_step(action)
                else:
                    obs, reward, done, truncated, info = self.step(action)
                total_reward += reward
                actions.append(action)

                if done:
                    break
                step += 1

            trajs[str(traj)] = {'states': np.array(
                states), 'actions': np.array(actions)}
            rewards.append(total_reward / steps)
            # print('Finished traj:', traj, 'Average Reward:', total_reward / steps)
        print("Avg Reward:", np.mean(rewards))
        if save:
            np.savez(os.path.join('experts', dir_name, file_name), env=dir_name,
                     num_trajs=num_trajs,
                     mean_reward=np.mean(rewards),
                     std_reward=np.std(rewards),
                     **trajs)

        return np.mean(rewards)



class HopperWrapper(gym.Wrapper):
    def __init__(self, env, uo_vis=False, ue_k=3, uo_k=10, uo_low=0.1, uo_high=3., ue_mu=0., ue_sigma=0.2,
                 action_noise=1., control_loss=0.1):
        super().__init__(env)
        # self.sticky = np.random.uniform(0.1, 2.0)
        self.ue_k = ue_k
        self.uo_k = uo_k
        self.uo_low = uo_low
        self.uo_high = uo_high
        self.ue_mu = ue_mu
        self.ue_sigma = ue_sigma
        self.action_noise = action_noise
        self.control_loss = control_loss

        self.env = env
        self.env.reset()
        self.uo_vis = uo_vis
        print(self.env.observation_space)  # 12 with pos 11 without
        self.dt = self.env.unwrapped.dt
        if self.uo_vis:
            self.observation_space = Box(low=-np.inf * np.ones(env.observation_space.shape[0] + 1),
                                         high=np.inf * np.ones(env.observation_space.shape[0] + 1),
                                         dtype=np.float32)

    def get_sticky(self):
        return np.clip(np.average(self.uo_queue), 0.1, 2)

    def reset(self, seed=666, **kwargs):
        self.uo_queue = np.random.uniform(self.uo_low, self.uo_high, self.uo_k)
        self.ue_queue = np.random.normal(self.ue_mu, np.sqrt(self.ue_k) * self.ue_sigma, self.ue_k)
        self.sticky = self.get_sticky()
        self.ue = np.average(self.ue_queue)
        obs, info = self.env.reset()
        if self.uo_vis:
            obs = np.concatenate([obs, [self.sticky]])
        # else:
        #     obs = np.concatenate([obs])
        return obs.astype(np.float32), info

    def confounded_step(self, action):
        cf_action = action + self.action_noise * self.ue
        next_obs, reward, done, truncated, info = self.step(action)
        next_obs[1:] += self.ue
        return next_obs, reward, cf_action, done, truncated, info

    def step(self, action):
        self.sticky = self.get_sticky()
        self.uo_queue = np.append(self.uo_queue[1:], np.random.uniform(self.uo_low, self.uo_high, 1))
        self.ue_queue = np.append(self.ue_queue[1:],
                                  np.random.normal(self.ue_mu, np.sqrt(self.ue_k) * self.ue_sigma, 1))
        self.ue = np.average(self.ue_queue)

        p = self.env.unwrapped.data.qpos[0]
        next_obs, _, done, truncated, info = self.env.step(action)
        p_prime = self.env.unwrapped.data.qpos[0]
        p_prime = p + (p_prime - p) / self.sticky
        # print('new p and stickness',p_prime,self.sticky)
        next_obs[0] = p_prime
        self.env.unwrapped.set_state(next_obs[:6], next_obs[6:])
        vel = (p_prime - p) / (self.dt)

        if self.uo_vis:
            next_obs = np.concatenate([next_obs, [self.sticky]])

        forward_reward = -1.0 * 1 * (vel - 1) ** 2
        ctrl_cost = self.control_loss * 1e-1 * np.sum(np.square(action))
        living_reward = 1  # for expert training
        reward = living_reward + forward_reward - ctrl_cost

        return next_obs.astype(np.float32), reward, done, truncated, info

    def evaluate_expert(self, expert, steps=500, episodes=100, confounded=False):
        total_reward = 0
        total_error = 0

        sum = 0
        for traj in range(episodes):
            obs, _ = self.reset()
            for i in range(steps):
                action, _state = expert.predict(obs, deterministic=True)
                sum += np.mean(np.square(action))
                if confounded:
                    cf_action = action + self.action_noise * self.ue
                    obs_new, r, done, truncated, info = self.step(cf_action)
                    obs_new += self.ue
                    obs=obs_new


                    # obs, r,action, _, _, _ = self.confounded_step(action)
                else:
                    obs, r, done, _, _ = self.step(action)
                    cf_action = action

                total_reward += r
                total_error += np.mean((action - cf_action) ** 2)

                if done:
                    break

        print(sum / (steps * episodes), 'norm of expert')
        return total_reward / (steps * episodes), total_error / (steps * episodes)

    def evaluate(self, policy, expert, steps=500, episodes=100, confounded=False,scaler=None):
        policy.eval()
        total_reward = 0
        total_error = 0
        total_steps=0
        for traj in range(episodes):
            obs, _ = self.reset()

            # loc=obs[0]
            # obs[0]=-0.05*100
            for step in range(steps):
                exp_act = expert.predict(obs, deterministic=True)[0]
                if scaler is not None:
                    scaled_obs=scaler.transform(obs.reshape(1,-1)).flatten()
                else:
                    scaled_obs=obs
                action = policy(torch.tensor(scaled_obs[:-1], dtype=torch.float32).to(device)).numpy(force=True)

                if confounded:
                    x, r, action,done, _, _ = self.confounded_step(action)
                else:
                    x, r, done, _, _ = self.step(action)
                obs = x
                total_reward += r
                total_error += np.mean((exp_act - action) ** 2)
                if done:
                    break
            total_steps+=step+1
        print(total_steps)
        return total_reward / total_steps, total_error / total_steps

    def dfiv_evaluate(self, policy, expert, policy_history=3, steps=500, episodes=100,
                      confounded=False, state_scaler=None, action_scaler=None):
        # policy.eval()
        total_reward = 0
        total_error = []
        for traj in range(episodes):
            obs, _ = self.reset()
            history = [torch.tensor(obs[:-1])] * policy_history

            for _ in range(steps):
                # print(torch.cat(history,dim=-1).shape)
                hist = torch.cat(history, dim=-1)
                hist[0] = (hist[0] - hist[12]) * 20
                hist = hist[:12]
                if state_scaler:
                    hist = state_scaler.transform(np.expand_dims(hist, 0))

                action = policy(torch.tensor(hist))
                if action_scaler:
                    action = action_scaler.inverse_transform(action)
                action = np.squeeze(action)
                exp_act = expert.predict(obs, deterministic=True)[0]
                # print(action,exp_act)
                if confounded:
                    x, r, action,done, _, _ = self.confounded_step(action)
                else:
                    x, r, done, _, _ = self.step(action)
                obs = x
                total_reward += r
                # total_error += np.mean((exp_act - action) ** 2)
                total_error.append((exp_act - action) ** 2)
                if done:
                    break

                history.pop(-1)
                history.insert(0, torch.tensor(obs[:-1]))
        err = np.mean(np.array(total_error), 0)

        return total_reward / (steps * episodes), err

    def history_evaluate(self, policy, expert, policy_history=3, steps=500, episodes=100, use_action=False,
                         confounded=False,scaler=None):
        # policy.eval()
        total_reward = 0
        total_error = 0
        total_steps=0
        for traj in range(episodes):
            obs, _ = self.reset()
            # if use_action:
            #     history = [torch.tensor(obs[:-1]),
            #                torch.zeros(self.env.action_space.shape, dtype=torch.float32)] * policy_history
            # else:
            if scaler is not None:
                obs = scaler.transform(obs.reshape(1, -1)).flatten()
            history = [torch.tensor(obs[:-1])] * policy_history

            for step in range(steps):
                # print(torch.cat(history,dim=-1).shape)
                exp_act = expert.predict(obs, deterministic=True)[0]

                hist = torch.cat(history, dim=-1).to(device)
                hist[0] = (hist[0] - hist[12] - 0.0) * 20
                hist = hist[:12]
                action = policy(hist).cpu()
                if confounded:
                    x, r, _, done, _, _ = self.confounded_step(action.numpy(force=True))
                else:
                    x, r, done, _, _ = self.step(action.numpy(force=True))
                obs = x
                total_reward += r
                total_error += np.mean((exp_act - action.numpy(force=True)) ** 2)
                # total_error.append((exp_act - action.numpy(force=True)) ** 2)

                if done:
                    break

                if use_action:
                    history.pop(-1)
                    history.insert(0, action)

                history.pop(-1)
                if scaler is not None:
                    obs=scaler.transform(obs.reshape(1,-1)).flatten()
                history.insert(0, torch.tensor(obs[:-1]))

            total_steps+=step+1

        print(total_steps)
        return total_reward / total_steps, total_error / total_steps


        err = np.mean(np.array(total_error), 0)
        return total_reward / (steps * episodes), err

    def gen_expert_demos(self, dir_name, file_name, model, num_trajs, steps, save=True, confounded=False):
        trajs = dict()
        rewards = []
        for traj in range(num_trajs):
            total_reward = 0
            obs, _ = self.reset()
            done = False
            states = []
            actions = []
            step = 0
            while not done and step < steps:
                states.append(obs)  # masking context
                action, _state = model.predict(obs, deterministic=True)
                # print(action)
                if confounded:
                    obs, reward, action, done, truncated, info = self.confounded_step(action)
                else:
                    obs, reward, done, truncated, info = self.step(action)
                total_reward += reward
                actions.append(action)

                if done:
                    break
                step += 1

            trajs[str(traj)] = {'states': np.array(
                states), 'actions': np.array(actions)}
            rewards.append(total_reward / steps)
            # print('Finished traj:', traj, 'Average Reward:', total_reward / steps)
        print("Avg Reward:", np.mean(rewards))
        if save:
            np.savez(os.path.join('experts', dir_name, file_name), env=dir_name,
                     num_trajs=num_trajs,
                     mean_reward=np.mean(rewards),
                     std_reward=np.std(rewards),
                     **trajs)

        return np.mean(rewards)