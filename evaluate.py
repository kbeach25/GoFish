# ChatGPT script with infinite loop guard
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from GoFishEnv import GoFishEnv
import numpy as np

# === Config ===
NUM_GAMES = 10000
SHOW_GAME_SUMMARY = True  # Set to False to disable per-game logs
MAX_STEPS_PER_GAME = 500  # Safeguard to skip potential infinite loops

# === Load model ===
model = PPO.load("GoFish_Model_easy")

# === Stats tracking ===
wins = 0
losses = 0
ties = 0
zero_games = 0
skipped_games = 0

for game in range(NUM_GAMES):
    env = FlattenObservation(GoFishEnv())
    obs, _ = env.reset()
    done = False
    step_count = 0

    while not done:
        action, _ = model.predict(obs)

        # Convert numpy array action to integer
        if isinstance(action, np.ndarray):
            action = int(action.item())

        obs, reward, done, truncated, info = env.step(action)
        step_count += 1

        if step_count >= MAX_STEPS_PER_GAME:
            print(f"Game {game+1}: Skipped due to exceeding {MAX_STEPS_PER_GAME} steps.")
            skipped_games += 1
            break

    if step_count >= MAX_STEPS_PER_GAME:
        continue  # Skip stats collection and move to next game

    agent_sets = sum(env.unwrapped.agent_sets)
    opponent_sets = sum(env.unwrapped.opponent_sets)

    if agent_sets == 0 and opponent_sets == 0:
        zero_games += 1
        result = "NO PROGRESS"
    elif agent_sets > opponent_sets:
        wins += 1
        result = "WIN"
    elif agent_sets < opponent_sets:
        losses += 1
        result = "LOSS"
    else:
        ties += 1
        result = "TIE"

    if SHOW_GAME_SUMMARY:
        print(f"Game {game+1}: {result} (Agent: {agent_sets}, Opponent: {opponent_sets}, Steps: {step_count})")

# === Final Report ===
print("\n=== Evaluation Summary ===")
print(f"Total Games Attempted: {NUM_GAMES}")
print(f"Wins:                  {wins}")
print(f"Losses:                {losses}")
print(f"Ties:                  {ties}")
print(f"No Progress Games:     {zero_games}")
print(f"Skipped Games:         {skipped_games}")
print(f"Win Rate:              {wins / (NUM_GAMES - skipped_games):.2%}" if NUM_GAMES != skipped_games else "Win Rate: N/A")
