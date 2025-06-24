from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from gymnasium.spaces.utils import flatten
from GoFishEnv import GoFishEnv

RANK_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

# Load the trained model
model = PPO.load("GoFish_Model")

# Create and wrap the environment
base_env = GoFishEnv(mode="play")
env = FlattenObservation(base_env)

obs, _ = env.reset()
done = False

print("Welcome to Go Fish! You're playing against a trained agent.\n")

while not done:
    # Check whose turn it is using the original (unwrapped) env
    is_agent_turn = base_env.agent_turn

    if is_agent_turn:
        # Show the human's hand nicely
        hand_counts = base_env.agent_hand
        print("\nYour hand:")
        shown = []
        for i in range(13):
            count = base_env.agent_hand.count(i)
            if count > 0:
                print(f"- {RANK_LABELS[i]}: {count} card(s)")
                shown.append(i)

        try:
            action = int(input(f"What rank do you want to ask for? (0–12) — Valid choices: {shown}: "))
            if action not in shown:
                raise ValueError
        except:
            print("❌ Invalid input — try again.")
            continue

    else:
        # It's the model's turn — get its action
        flat_obs = flatten(env.observation_space, obs)
        action, _ = model.predict(flat_obs, deterministic=True)
        print(f"\n🤖 Opponent asked for: {RANK_LABELS[action]}")

    # Step with the selected action
    obs, reward, done, _, info = env.step(action)

# Final score
print("\n🎉 Game over!")
print(f"Your sets: {base_env.agent_sets.count(1)}")
print(f"AI sets:   {base_env.opponent_sets.count(1)}")
