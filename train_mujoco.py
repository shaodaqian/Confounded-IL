import gymnasium as gym
# import pybullet_envs
from stable_baselines3 import PPO, SAC
from torch import nn

from models.utils import device
from envs import AntWrapper,HCWrapper,HopperWrapper

def train_halfcheetah(HC_env):

    HC_env.reset()
    SAC_model = SAC('MlpPolicy', HC_env, verbose=1,
                    buffer_size=300000, batch_size=256, gamma=0.98, tau=0.02,
                    train_freq=64, gradient_steps=64, ent_coef='auto', learning_rate=7.3e-4,
                    learning_starts=30000, policy_kwargs=dict(net_arch=[512, 512], log_std_init=-3),
                    use_sde=True,device=device)
    SAC_model.learn(total_timesteps=1000000)
    SAC_model.save("experts/HalfCheetah-v4/SAC_half_cheetah")



def train_ant(Ant_env):
    Ant_env.reset()

    PPO_model = PPO("MlpPolicy", Ant_env, verbose=1,
                batch_size=64,gae_lambda=0.8, gamma=0.98, learning_rate=3e-05,ent_coef=4.9646e-07,
                n_epochs=10, clip_range=0.1, n_steps=512,
                policy_kwargs=dict(log_std_init=-2, ortho_init=False, activation_fn=nn.ReLU, net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                max_grad_norm=0.6, vf_coef=0.6, normalize_advantage=True,device=device,
                # use_sde=True, sde_sample_freq=4,
                    )
    PPO_model.learn(total_timesteps=1e7)
    PPO_model.save("experts/Ant-v4/PPO_ant_expert")
    HC_PPO = PPO.load("experts/Ant-v4/PPO_ant_expert")
    print('Ant-v4 expert trained')
    expert_reward = Ant_env.evaluate_expert(HC_PPO, episodes=100)
    print(f"PPO 2 Expert reward: {expert_reward}")



def train_hopper(Hopper_env):
    Hopper_env.reset()
    SAC_model = SAC('MlpPolicy', Hopper_env, verbose=1,
                    buffer_size=1000000, batch_size=256, gamma=0.99, tau=0.005,
                    train_freq=1, gradient_steps=1, ent_coef='auto', learning_rate=0.0003,
                    learning_starts=10000, policy_kwargs=dict(net_arch=[512, 512], log_std_init=-3),
                    use_sde=False,device=device)

    SAC_model.learn(total_timesteps=3e6)
    SAC_model.save("experts/Hopper-v4/SAC_hopper_expert")
    hopper_sac = SAC.load("experts/Hopper-v4/SAC_hopper_expert")
    print('Hopper-v4 expert trained')
    expert_reward = Hopper_env.evaluate_expert(hopper_sac, episodes=100)
    print(f"SAC Expert reward: {expert_reward}")

if __name__=="__main__":
    ue_k=0
    uo_k=20

    hopper_env = HopperWrapper(gym.make("Hopper-v4", exclude_current_positions_from_observation=False),
                       uo_vis=True,
                       ue_k=ue_k,
                       uo_k=uo_k,
                       uo_low=-2,
                       uo_high=4,
                       ue_mu=0.,
                       ue_sigma=0.0,
                       action_noise=0,
                       control_loss=0.2)
    hopper_env.reset()
    # Ant_env.step(Ant_env.action_space.sample())
    train_hopper(hopper_env)