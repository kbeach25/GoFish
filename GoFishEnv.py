import gymnasium as gym 
from gymnasium import spaces
from gymnasium.spaces.utils import flatten
import numpy as np
import random

class GoFishEnv(gym.Env):
    def __init__(self, mode="train"):
        # Initialize environment
        super().__init__()

        # Initialize human player
        self.mode = mode

        # Change 'who' the model is in the gameplay depending on mode
        self.model_role = "agent" if mode == "train" else "opponent"

        # Define action and observation spaces
        self.action_space = spaces.Discrete(13) # 13 possible ranks

        # Keep track of ask failures to avoid mistakes
        self.recent_failed_asks = {}
        self.turn_counter = 0
        

        # Observations space,  multibinary used for a space of binary values
        self.observation_space = spaces.Dict({
            "agent_hand_ranks": spaces.MultiDiscrete([5]*13),
            "opponent_hand_size": spaces.Discrete(53),
            "agent_sets_completed": spaces.Discrete(14),
            "opponent_sets_completed": spaces.Discrete(14),
            "is_agent_turn": spaces.Discrete(2),
            "last_agent_ask": spaces.Discrete(14), # 0-12, 13 for no ask yet
            "last_agent_ask_success": spaces.Discrete(2), # Yes or no
            "last_opponent_ask": spaces.Discrete(14),
            "last_opponent_ask_success": spaces.Discrete(2)
            })

        # Define game environment
        self.deck = self._init_deck()
        self.agent_hand = []
        self.opponent_hand = [] 
        self.agent_sets = [0] * 13 # Completed sets of 4 cards
        self.opponent_sets = [0] * 13
        self.turn = 0

        # Give the agent recent memory
        self.last_agent_ask = 13
        self.last_agent_ask_success = 0
        self.last_opponent_ask = 13
        self.last_opponent_ask_success = 0

        
    def reset(self, seed=None, options=None):
        # Reset game to default environment
        
        # Shuffle deck
        self.deck = self._init_deck()
        random.shuffle(self.deck)

        # Determine who goes first with coin flip
        coin_flip = random.randint(0, 1)
        first = 0
        if coin_flip == 1:
            first = 1

        self.coin_flip_result = coin_flip

        self.agent_turn = True if first == 0 else False

            
        # Reset agent and oppponent hands
        self.agent_hand = []
        self.opponent_hand = []

        # Reset failed asks and turns
        self.recent_failed_asks = {}
        self.turn_counter = 0
        
        # Deal hands
        dealt = 0
        while dealt < 14:
            # Deal to agent first if coin flip is 0
            if coin_flip == 0:
                self.agent_hand.append(self.deck.pop())
                self.opponent_hand.append(self.deck.pop())
                dealt += 2
            else:
                self.opponent_hand.append(self.deck.pop())
                self.agent_hand.append(self.deck.pop())
                dealt += 2

        # Reset agent and opponent completed sets, turns
        self.agent_sets = [0]*13
        self.opponent_sets = [0]*13
        self.turn = 0

        # Reset memory
        self.last_agent_ask = 13
        self.last_agent_ask_success = 0
        self.last_opponent_ask = 13
        self.last_opponent_ask_success = 0

        # Set initial observations
        obs = self._get_observation()
        return obs, {}

    # Step function, function differs depending on mode
    def step(self, action):
        if self.mode == "train":
            return self.training_step(action)
        elif self.mode == "play":
            return self.step_play(action)
        else:
            print("This should be unreachable, line 104 of GoFishEnv")

    def step_play(self, action, obs=None):
        if obs is None:
            obs = self._get_observation()
        # Human turn
        if self.agent_turn:
            if not self._can_ask(action, player="agent"):
                self.agent_turn = False
                return self._get_observation(), -0.1, False, False, {"reason": "invalid_action"}

            success = self._process_ask(action, player="agent")
            self._update_sets()

            # track human ask
            self.last_agent_ask = action
            self.last_agent_ask_success = int(success)
              
            reward = 0.01
            if success:
                self._check_empty_hand()
                reward += 0.3 * self.opponent_hand.count(action)
                if self.agent_hand.count(action) == 1:
                    reward -= 0.5

            else:
                # Go fish if unsuccessful ask
                if self.deck:
                    self.agent_hand.append(self.deck.pop())
                    reward -= 0.05
                self.agent_turn = False

            done = self._check_game_over()
            return self._get_observation(), reward, done, False, {}

        # Agent acting as opponent for play mode
        else:
            if hasattr(self, "model") and self.model is not None:
                opponent_obs = self._get_opponent_observation()
                flat_obs = flatten(self.observation_space, opponent_obs)
                action, _ = self.model.predict(flat_obs, deterministic=True)
                valid_asks = [rank for rank in range(13) if rank in self.opponent_hand]
                # If agent attempts illegal move
                if action not in valid_asks:
                    if valid_asks:
                        action = random.choice(valid_asks)
                    else:
                        action = 0
                        
                print(f'\nOpponent asked for: {action}')
                
            else: # Fall back
                valid_asks = [rank for rank in range(13) if rank in self.opponent_hand]
                # If agent attempts illegal move
                if action not in valid_asks:
                    if valid_asks:
                        action = random.choice(valid_asks)
                    else:
                        action = 0
                print(f'\nOpponent asked for {action} via fallback')
                
            success = self._process_ask(action, player="opponent")
            self._update_sets()
            self._check_empty_hand()

            self.last_opponent_ask = action
            self.last_opponent_ask_success = int(success)

            if success:
                done = self._check_game_over()
                return self._get_observation(), 0, done, False, {}
            else:
                if self.deck:
                    self.opponent_hand.append(self.deck.pop())

                self.agent_turn = True
                done = self._check_game_over()
                return self._get_observation(), 0, done, False, {}
        


    # Function for training model, not for play
    def training_step(self, action): # Must return observation, reward, terminated, truncated, info
        # Action is just asking do you have rank 2, 3, etc
        
        # Make sure agent doesn't act out of turn, go to opponent turn if they do
        if not self.agent_turn or not self._can_ask(action):
            reward = -1.0
            reason = "moved_out_of_turn" if not self.agent_turn else "invalid_action"
            
            self.agent_turn = False

            # Opponent turn, asks for whatever card it has the most of or a random card
            while self.opponent_hand and not self._check_game_over():
                
                counts = [self.opponent_hand.count(r) for r in range(13)]
                opponent_rank = int(np.argmax(counts))

                # 30% chance of a random ask, trying to make opponent less predictable 
                random_move = random.uniform(1, 100)
                if random_move <= 30:
                    opponent_rank = random.choice(counts)
                    
                success = self._process_ask(opponent_rank, player="opponent")
                self._update_sets()

                self.last_opponent_ask = opponent_rank
                self.last_opponent_ask_success = int(success)

                # Draw additional card if someone runs out of cards during turn
                self._check_empty_hand()
                

                if not success:
                    # Opponent go fish
                    if self.deck:
                        self.opponent_hand.append(self.deck.pop(0))
                    break

            self.agent_turn = True
            done = self._check_game_over()

            return self._get_observation(), reward, done, False, {"reason": reason}

        self.turn_counter += 1

        recent_fail_penalty = 0
        if action in self.recent_failed_asks:
            recent_failures = [turn for turn in self.recent_failed_asks[action] if self.turn_counter - turn <= 5]

            if recent_failures:
                recent_fail_penalty = -0.2 * len(recent_failures)

        reward = 0.01
        reward += recent_fail_penalty
            
        prev_sets = sum(self.agent_sets)
        opp_hand_prev = len(self.opponent_hand)
        success = self._process_ask(action, player="agent")
        self._update_sets()

        self.last_agent_ask = action
        self.last_agent_ask_success = int(success)

        # Draw cards if hand is empty
        self._check_empty_hand()

        # Reward based on how many new cards are receievd and new sets 
        updated_sets = sum(self.agent_sets)
        new_sets = updated_sets - prev_sets
        new_cards = opp_hand_prev - len(self.opponent_hand)

        # Reward of 1 per new set, 0.3 per new card
        if success:
            reward += 0.3 * new_cards
        reward += 1 * new_sets
            
        # See if move ended game
        done = self._check_game_over()

        if done:
            return self._get_observation(), reward, True, False, {}

        if success: # Agent goes again if successful turn
            self.agent_turn = True
            return self._get_observation(), reward, False, False, {}

        # Discourage blind guessing
        if success and self.agent_hand.count(action) == 1:
            reward -= 0.5

        # Go fish if unsuccessful ask
        if not success:
            if action not in self.recent_failed_asks:
                self.recent_failed_asks[action] = []
            self.recent_failed_asks[action].append(self.turn_counter)

            if len(self.recent_failed_asks[action]) > 10:
                   self.recent_failed_asks[action] = self.recent_failed_asks[action][-10:]
                   
            if self.deck:
                self.agent_hand.append(self.deck.pop())
                reward -= 0.05
                   
            self.agent_turn=False

        # Opponent turn, asks for whatever card it has the most of 
        while self.opponent_hand and not self._check_game_over():
            # opponent_rank = random.choice(list(set(self.opponent_hand)))
            counts = [self.opponent_hand.count(r) for r in range(13)]
            opponent_rank = int(np.argmax(counts))
            success = self._process_ask(opponent_rank, player="opponent")
            self._update_sets()
            self._check_empty_hand()

            self.last_opponent_ask = opponent_rank
            self.last_opponent_ask_success = int(success)

            if not success:
                # Opponent go fish
                if self.deck:
                    self.opponent_hand.append(self.deck.pop(0))
                break
        self.agent_turn = True

        done = self._check_game_over()

        if self.turn_counter % 5 == 0:
            self._remove_old_fails()
                   
        return self._get_observation(), reward, done, False, {}

        

    # Deck building helper function, only need ranks
    def _init_deck(self):
        return [rank for rank in range(13) for _ in range(4)]

    # Get observation helper function, 5 observations
    def _get_observation(self):
        
        # Hand vector to count ranks in hand
        hand_vector = [self.agent_hand.count(rank) for rank in range(13)]

        # Assign hand vector to hand_ranks
        obs = {
            "agent_hand_ranks": hand_vector,
            "opponent_hand_size": len(self.opponent_hand),
            "agent_sets_completed": sum(self.agent_sets),
            "opponent_sets_completed": sum(self.opponent_sets),
            "is_agent_turn": int(self.agent_turn),
            "last_agent_ask": self.last_agent_ask,
            "last_agent_ask_success": self.last_agent_ask_success,
            "last_opponent_ask": self.last_opponent_ask,
            "last_opponent_ask_success": self.last_opponent_ask_success
            }
        
        return obs

    def _can_ask(self, rank, player="agent"):
        if player == "agent":
            # Verify legal moves with True or False
            return rank in self.agent_hand
        else:
            return rank in self.opponent_hand

    def _process_ask(self, rank, player):
        # Agent or opponent asks for a rank
        # They either get those cards (True) or draw cards from deck (False)
        
        if player == "agent": # Agent Case
            asker = self.agent_hand
            target = self.opponent_hand
            
        else: # Random Opponent Case
            asker = self.opponent_hand
            target = self.agent_hand

        
        opponent_cards_of_rank = [card for card in target if card == rank]

        # Turn over all cards of rank if True
        if opponent_cards_of_rank:
            for card in opponent_cards_of_rank:
                asker.append(card)
                target.remove(card)
            return True

        else:
            return False

    # Function to note complete sets 
    def _update_sets(self):
        # Check each index to see if set is complete, remove from hand if yes
        for rank in range(13):
            if self.agent_hand.count(rank) == 4 and self.agent_sets[rank] == 0:
                self.agent_sets[rank] = 1
                self.agent_hand = [card for card in self.agent_hand if card != rank]

        # Repeat for opponent
        for rank in range(13):
            if self.opponent_hand.count(rank) == 4 and self.opponent_sets[rank] == 0:
                self.opponent_sets[rank] = 1
                self.opponent_hand = [card for card in self.opponent_hand if card != rank]
                

    def _check_game_over(self): # True if all sets have been completed
        sets = sum(self.agent_sets) + sum(self.opponent_sets)
        return sets == 13

    # Model assignment
    def set_model(self, model):
        self.model = model

    def _get_opponent_observation(self):
        # Reverse perspective of observations
        opponent_hand_vector = [self.opponent_hand.count(rank) for rank in range(13)]
        
        obs = {
            "agent_hand_ranks": opponent_hand_vector,  # Opponent's hand from their perspective
            "opponent_hand_size": len(self.agent_hand),  # Human player's hand size
            "agent_sets_completed": sum(self.opponent_sets),  # Opponent's sets
            "opponent_sets_completed": sum(self.agent_sets),  # Human's sets
            "is_agent_turn": 1,  # It's the opponent's turn, so from their perspective it's their turn
            # Previous move memory gets flipped in this case
            "last_agent_ask": self.last_opponent_ask,
            "last_agent_ask_success": self.last_opponent_ask_success,
            "last_opponent_ask": self.last_agent_ask,
            "last_opponent_ask_success": self.last_agent_ask_success
        }
        
        return obs

    # Handle empty hand, skip turn if no cards left in deck (shouldn't happen)
    def _check_empty_hand(self):
        if self.agent_turn:
            if not self.agent_hand:
                if self.deck:
                    self.agent_hand.append(self.deck.pop())
                else:
                    self.agent_turn = False

        else:
            if not self.opponent_hand:
                if self.deck:
                    self.opponent_hand.append(self.deck.pop())
                else:
                    self.agent_turn = True

    def _remove_old_fails(self):
        for rank in list(self.recent_failed_asks.keys()):
            self.recent_failed_asks[rank] = [
                turn for turn in self.recent_failed_asks[rank]
                if self.turn_counter - turn <= 10
            ]
            if not self.recent_failed_asks[rank]:
                del self.recent_failed_asks[rank]
