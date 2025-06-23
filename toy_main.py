import logging
import pandas as pd
import time
import sys

from toylearner import *
from envs import NoisyDroneEnv




if __name__ == '__main__':

    logging.basicConfig(filename="logs.log", filemode="a", format="%(asctime)s - %(levelname)s: %(message)s",level=logging.INFO)

    # logging.warning("warning1")
    logging.info("start")

    repeats=10
    results = []

    for policy_history in [3]:
        for ue_k in [20]:
            for i in range(repeats):
                epoch=100
                uo_k=30
                # policy_history=5
                print(f"ue_k: {ue_k}, uo_k: {uo_k}, policy_history: {policy_history}")
                logging.info(f"ue_k: {ue_k}, uo_k: {uo_k}, policy_history: {policy_history}")

                drone_env=NoisyDroneEnv(ue_k=ue_k,uo_k=uo_k,x0=(1,0),uo_low=-1.0,uo_high=1.0,ue_mu=0.,ue_sigma=0.1,action_noise=10)
                dataset=drone_env.get_dataset(20000)
                prev_states=dataset[:,0]
                actions=dataset[:,1]

                expert_reward,expert_mse=drone_env.evaluate(policy=drone_env.expert_policy)
                print(f"expert reward: {expert_reward},expert mse: {expert_mse}")
                logging.info(f"expert reward: {expert_reward},expert mse: {expert_mse}")
                results.append([ue_k,uo_k,policy_history,'expert',expert_reward,expert_mse])

                naive_policy=naive_learner(prev_states,actions,epoch,verbose=False)
                naive_reward,naive_mse=drone_env.evaluate(policy=lambda x,u1,u2: naive_policy(torch.tensor(x).float().view(-1,1).to(device)).item())
                print(f"Naive policy reward: {naive_reward}, naive mse: {naive_mse}")
                logging.info(f"Naive policy reward: {naive_reward}, naive mse: {naive_mse}")
                results.append([ue_k,uo_k,policy_history,'naive_policy',naive_reward,naive_mse])

                history_policy=history_learner(prev_states,actions,epoch,policy_history=policy_history,verbose=False)
                history_reward,history_mse=drone_env.history_evaluate(policy=history_policy,policy_history=policy_history)
                print(f"History policy reward: {history_reward}, history mse: {history_mse}")
                logging.info(f"History policy reward: {history_reward}, history mse: {history_mse}")
                results.append([ue_k,uo_k,policy_history,'history_policy',history_reward,history_mse])

                IV_policy=IV_learner(prev_states,actions,epoch,ue_k=ue_k,verbose=False)
                IV_reward,IV_mse=drone_env.evaluate(policy=lambda x,u1,u2: IV_policy(torch.tensor(x).float().view(-1,1).to(device)).item())
                print(f"IV policy reward: {IV_reward}, IV mse: {IV_mse}")
                logging.info(f"IV policy reward: {IV_reward}, IV mse: {IV_mse}")
                results.append([ue_k,uo_k,policy_history,'IV_policy',IV_reward,IV_mse])

                IV_history_policy=IV_history_learner(prev_states,actions,epoch,ue_k=ue_k,sample_k=1,policy_history=policy_history,verbose=False)
                IV_history_reward,IV_history_mse=drone_env.history_evaluate(policy=IV_history_policy,policy_history=policy_history)
                print(f"IV history policy reward: {IV_history_reward}, IV history mse: {IV_history_mse}")
                logging.info(f"IV history policy reward: {IV_history_reward}, IV history mse: {IV_history_mse}")
                results.append([ue_k,uo_k,policy_history,'IV_history_policy',IV_history_reward,IV_history_mse])


                IV_history_policy=DMLIV_history_learner(prev_states,actions,epoch,ue_k=ue_k,sample_k=1,
                                                        policy_history=policy_history,k_fold=10,
                                                        verbose=False)
                IV_history_reward,IV_history_mse=drone_env.history_evaluate(policy=IV_history_policy,policy_history=policy_history)
                print(f"DMLIV history policy reward: {IV_history_reward}, IV history mse: {IV_history_mse}")
                logging.info(f"DML IV history policy reward: {IV_history_reward}, IV history mse: {IV_history_mse}")
                results.append([ue_k,uo_k,policy_history,'DMLIV_history_policy',IV_history_reward,IV_history_mse])

                sys.stdout.flush()


    results=pd.DataFrame(results,columns=['ue_k','uo_k','policy_history','method','reward','mse'])
    # results=np.array(results)
    print(results)
    results.to_csv(f"results/toy/exp_{ue_k}.csv", index=False)
