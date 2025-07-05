from GoFishEnv import GoFishEnv
import time
import random
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = GoFishEnv()
env = FlattenObservation(env)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=40_000)

model.save("GoFish_Model")


obs, info = env.reset()
done = False
step_count = 0
'''
print("Starting Game...\n")

while not done:
    # Keep track of steps
    step_count += 1

    valid_plays = [rank for rank in range(13) if obs['agent_hand_ranks'][rank]==1]
    # Empty hand, nothing to ask for
    if not valid_plays:
        action = env.action_space.sample()
    else:
        action = random.choice(valid_plays)
        
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"Step {step_count}")
    print(f"Agent asked for rank: {action}")
    print(f"Reward: {reward}")
    print(f"Agent sets completed: {obs['agent_sets_completed']}")
    print(f"Opponent sets completed: {obs['opponent_sets_completed']}")
    print(f"Opponent hand size: {obs['opponent_hand_size']}")
    print(f"Agent hand ranks: {[i for i, val in enumerate(obs['agent_hand_ranks']) if val == 1]}")
    print(f"Done: {done}")
    print("-" * 40)
    
    time.sleep(0.5)  # Pause for readability, optional

print("Game over!")



from GoFishEnv import GoFishEnv
env = GoFishEnv()
obs, _ = env.reset()
done = False

while not done:
    valid = [i for i, v in enumerate(obs['agent_hand_ranks']) if v == 1]
    action = valid[0] if valid else 0  # Always ask a legal rank
    obs, reward, done, _, info = env.step(action)

print("Final Sets:", obs['agent_sets_completed'])
'''
