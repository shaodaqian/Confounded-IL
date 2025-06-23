import time
from typing import Callable, Union
import sys

import gymnasium as gym
import pandas as pd
from sklearn.preprocessing import StandardScaler
# import pybullet_envs
from stable_baselines3 import PPO, SAC

from DFIV_trainer import DFIVTrainer
from envs import HCWrapper, AntWrapper, HopperWrapper
from learner import *
from models.utils import device


def train_halfcheetah():
    confounded = True
    repeats = 20
    results = []
    for ue_k in [20]:
        for uo_k in [20]:
            for i in range(repeats):
                HC_env = HCWrapper(gym.make("HalfCheetah-v4", exclude_current_positions_from_observation=False),
                                   uo_vis=True,
                                   ue_k=ue_k,
                                   uo_k=uo_k,
                                   uo_low=-2,
                                   uo_high=4,
                                   # uo_low=1.,
                                   # uo_high=1.,
                                   ue_mu=0.,
                                   ue_sigma=0.01,
                                   action_noise=15,
                                   control_loss=0)
                HC_env.reset()
                print('ue is ',ue_k)
                HC_SAC = SAC.load("experts/HalfCheetah-v4/SAC_varUo_halfcheetah_expert",device=device)
                print('HalfCheetah expert loaded')

                expert_reward, expert_mse = HC_env.evaluate_expert(HC_SAC, episodes=50, steps=500,
                                                                   confounded=confounded)
                print(f"SAC Expert reward: {expert_reward}, MSE: {expert_mse}")
                results.append(['Expert', expert_reward, expert_mse, ue_k, uo_k])

                # model2 = SAC.load("experts/AntBulletEnv-v0/ant_expert3")

                num_trajectories = 20  # total of 100
                exp_reward = HC_env.gen_expert_demos("HalfCheetah-v4", 'SAC_uo_ue_demos', HC_SAC,
                                                     num_trajs=num_trajectories, steps=500, save=True,
                                                     confounded=confounded)
                # gen_expert_demos("AntBulletEnv-v0", env2, model2, 100, save=True)
                demos = np.load("./experts/{0}/SAC_uo_ue_demos.npz".format("HalfCheetah-v4"), allow_pickle=True)
                # demos = np.load("./experts/{0}/SAC_feateng_full_demos.npz".format("HalfCheetah-v4"), allow_pickle=True)

                obs = []
                acts = []
                for traj in range(num_trajectories):
                    s = demos[str(traj)].item()['states']
                    obs.append(s)
                    acts.append(demos[str(traj)].item()['actions'])

                prev_states = np.concatenate(obs, axis=0).astype(np.float32)
                actions = np.concatenate(acts, axis=0)

                prev_states = prev_states[:, :-1]
                print('Expert demonstrations loaded')
                print(actions.shape, prev_states.shape)

                policy_history = ue_k + 2
                use_action = False
                verbose = False
                batch_size = 64
                lr = 0.0001
                epochs = 150
                dropout = 0
                weight_decay = 1e-5

                DMLIV_history_policy=DMLIV_history_learner(prev_states,actions,
                                                    n_epochs = epochs,
                                                    ue_k = ue_k,
                                                    policy_history=policy_history,
                                                    sample_k=1,
                                                    lr=lr,
                                                    batch_size=batch_size,
                                                    dropout=dropout,
                                                    weight_decay=weight_decay,
                                                    n_components=5,
                                                    n_samples=1,
                                                    use_action=use_action,
                                                    hiddens=[256,256],
                                                    verbose=False,
                                                    k_fold=10)
                DMLIV_history_reward, DMLIV_history_mse = HC_env.history_evaluate(DMLIV_history_policy, HC_SAC, policy_history=policy_history,episodes=50,use_action=use_action,confounded=confounded)
                print(f"DMLIV history policy reward: {DMLIV_history_reward}, DMLIV history mse: {DMLIV_history_mse}")
                results.append(['DMLIV_history_policy',DMLIV_history_reward, DMLIV_history_mse,ue_k, uo_k])

                IV_history_policy = IV_history_learner(prev_states, actions,
                                                       n_epochs=epochs,
                                                       ue_k=ue_k,
                                                       policy_history=policy_history,
                                                       sample_k=ue_k,
                                                       lr=lr,
                                                       batch_size=batch_size,
                                                       dropout=dropout,
                                                       weight_decay=weight_decay,
                                                       n_components=5,
                                                       n_samples=1,
                                                       use_action=use_action,
                                                       hiddens=[256, 256],
                                                       verbose=verbose, )

                IV_history_reward, IV_history_mse = HC_env.history_evaluate(IV_history_policy, HC_SAC,
                                                                            policy_history=policy_history, episodes=50,
                                                                            use_action=use_action,
                                                                            confounded=confounded)
                print(f"IV history policy reward: {IV_history_reward}, IV history mse: {IV_history_mse}")
                results.append(['IV_history_policy', IV_history_reward, IV_history_mse, ue_k, uo_k])

                IV_history_policy = IV_history_learner(prev_states, actions,
                                                       n_epochs=epochs,
                                                       ue_k=ue_k,
                                                       policy_history=policy_history,
                                                       sample_k=1,
                                                       lr=lr,
                                                       batch_size=batch_size,
                                                       dropout=dropout,
                                                       weight_decay=weight_decay,
                                                       n_components=5,
                                                       n_samples=1,
                                                       use_action=use_action,
                                                       hiddens=[256, 256],
                                                       verbose=verbose, )

                IV_history_reward, IV_history_mse = HC_env.history_evaluate(IV_history_policy, HC_SAC,
                                                                            policy_history=policy_history, episodes=50,
                                                                            use_action=use_action,
                                                                            confounded=confounded)
                print(f"1sample IV history policy reward: {IV_history_reward}, IV history mse: {IV_history_mse}")
                results.append(['1sample IV_history_policy', IV_history_reward, IV_history_mse, ue_k, uo_k])

                IV_policy = IV_learner(prev_states, actions,
                                       n_epochs=epochs,
                                       ue_k=ue_k,
                                       lr=lr,
                                       batch_size=batch_size,
                                       dropout=dropout,
                                       weight_decay=weight_decay,
                                       hiddens=[256, 256],
                                       n_components=5,
                                       n_samples=1,
                                       verbose=verbose, )
                IV_reward, IV_mse = HC_env.evaluate(IV_policy, HC_SAC, episodes=50, confounded=confounded)
                print(f"IV policy reward: {IV_reward}, IV mse: {IV_mse}")
                results.append(['IV_policy', IV_reward, IV_mse, ue_k, uo_k])

                history_policy = history_learner(prev_states, actions, n_epochs=epochs, policy_history=policy_history, hiddens=[256,256], lr=lr,
                                                 dropout=dropout, weight_decay=weight_decay, batch_size=batch_size,use_action=use_action,verbose=verbose)
                history_reward, history_mse = HC_env.history_evaluate(history_policy, HC_SAC, policy_history=policy_history,episodes=50,use_action=use_action,confounded=confounded)
                print(f"History policy reward: {history_reward}, history mse: {history_mse}")
                results.append(['history_policy',history_reward,history_mse,ue_k,uo_k])

                naive_policy = naive_learner(prev_states, actions, n_epochs=epochs, hiddens=[256, 256], lr=lr,
                                             dropout=dropout, weight_decay=weight_decay, batch_size=batch_size,
                                             verbose=verbose)
                naive_reward, naive_mse = HC_env.evaluate(naive_policy, HC_SAC, episodes=50, confounded=confounded)
                print(f"Naive policy reward: {naive_reward}, naive mse: {naive_mse}")
                results.append(['naive_policy', naive_reward, naive_mse, ue_k, uo_k])

                sys.stdout.flush()

    results = pd.DataFrame(results, columns=['method', 'reward', 'mse', 'ue_k', 'uo_k'])
    timestr = time.strftime("%m%d-%H")
    print(results)
    results.to_csv(f"results/half_cheetah/exp_{timestr}_{ue_k}.csv", index=False)



if __name__ == "__main__":
    train_halfcheetah()