from GoFishEnv import GoFishEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
import numpy as np

class LegalActionWrapper(FlattenObservation):
    def step(self, action):
        # Prevent illegal moves during training
        legal_ranks = [i for i, x in enumerate(self.env.agent_hand) if x == i]
        if not legal_ranks:
            action = 0
        elif action not in legal_ranks:
            action = np.random.choice(legal_ranks)

        return super().step(action)

env = LegalActionWrapper(GoFishEnv())
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("GoFish_Model")
