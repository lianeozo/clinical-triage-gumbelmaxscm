import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from MDP import MDP
from State import State
from Action import Action


# ---------------------------------------------------------
# GYM ENVIRONMENT WRAPPER
# ---------------------------------------------------------

class TriageEnv(gym.Env):

    def __init__(self, max_steps=20):

        super().__init__()

        self.max_steps = max_steps
        self.step_count = 0

        self.mdp = MDP(
            init_state_idx=None,
            policy_array=None,
            policy_idx_type="obs",
            p_diabetes=0.2
        )

        # observation = state vector from simulator
        self.observation_space = spaces.Box(
            low=0,
            high=5,
            shape=(11,),
            dtype=np.float32
        )

        # actions correspond to Action(action_idx)
        self.action_space = spaces.Discrete(Action.NUM_ACTIONS_TOTAL)


    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.mdp.state = self.mdp.get_new_state()
        self.step_count = 0

        obs = self.mdp.state.get_state_vector().astype(np.float32)

        return obs, {}


    # ---------------------------------------------------------
    # STEP
    # ---------------------------------------------------------

    def step(self, action_idx):

        action = Action(action_idx=int(action_idx))

        # enforce feasibility
        if not self.mdp.soc_feasibility(action.soc) or \
           not self.mdp.treatment_feasibility(action):

            reward = -0.2
        else:
            reward = self.mdp.transition(action)

        obs = self.mdp.state.get_state_vector().astype(np.float32)

        self.step_count += 1

        terminated = reward != 0
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}


# ---------------------------------------------------------
# TRAIN PPO
# ---------------------------------------------------------

def train():

    # parallel environments (much faster)
    env = make_vec_env(TriageEnv, n_envs=8)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        tensorboard_log="./ppo_triage_logs/"
    )

    model.learn(total_timesteps=1_000_000)

    model.save("ppo_triage_model")

    print("Training complete. Model saved.")


# ---------------------------------------------------------
# EVALUATE POLICY
# ---------------------------------------------------------

def evaluate():

    env = TriageEnv()

    model = PPO.load("ppo_triage_model")

    obs, _ = env.reset()

    total_reward = 0

    for step in range(50):

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, trunc, _ = env.step(action)

        total_reward += reward

        if done or trunc:
            break

    print("Episode reward:", total_reward)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    train()

    evaluate()