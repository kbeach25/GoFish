from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from gymnasium.spaces.utils import flatten
from GoFishEnv import GoFishEnv

RANK_LABELS = [str(i) for i in range(13)]  # '0' to '12'

# Load the trained model
model = PPO.load("GoFish_Model")

# Create and wrap the environment
base_env = GoFishEnv(mode="play")
base_env.set_model(model)
env = FlattenObservation(base_env)

obs, _ = env.reset()
done = False

print("Welcome to Go Fish! You're playing against a trained agent.\n")

while not done:
    # Always show both hands
    print("\n=== Current Hands ===")

    print("Your hand:")
    player_shown = []
    for i in range(13):
        count = base_env.agent_hand.count(i)
        if count > 0:
            print(f"- Rank {i}: {count} card(s)")
            player_shown.append(i)

    print("\nOpponent's hand:")
    opp_counts = []
    for i in range(13):
        count = base_env.opponent_hand.count(i)
        if count > 0:
            print(f"- Rank {i}: {count} card(s)")
            opp_counts.append((i, count))

    print("\n======================")

    is_agent_turn = base_env.agent_turn

    if is_agent_turn:
        try:
            action = int(input(f"What rank do you want to ask for? (0–12) — Valid choices: {player_shown}: "))
            if action not in player_shown:
                raise ValueError
        except:
            print("❌ Invalid input — try again.")
            continue

        obs, reward, done, _, info = env.step(action)

    else:
        # Opponent's turn — model predicts
        action, _ = model.predict(obs, deterministic=True)
        # print(f"\n🤖 Opponent asked for: {action}")

        # Step with the selected action
        obs, reward, done, _, info = env.step(action)

# Final score
print("\n🎉 Game over!")
print(f"Your sets: {base_env.agent_sets.count(1)}")
print(f"AI sets:   {base_env.opponent_sets.count(1)}")
