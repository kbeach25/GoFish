import streamlit as st
import requests
import random
from PIL import Image
from io import BytesIO
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from gymnasium.spaces.utils import flatten
from GoFishEnv import GoFishEnv
import time

# Create and wrap environment
env = GoFishEnv(mode='play')

# Function to get a new deck from deckofcardsapi
def getDeck():
    # Pull deck from website
    url = "https://deckofcardsapi.com/api/deck/new/shuffle/?deck_count=1"
    response = requests.get(url).json()
    id = response['deck_id']

    # Use a stack to store cards
    draw_url = f"https://deckofcardsapi.com/api/deck/{id}/draw/?count=52"
    draw_response = requests.get(draw_url).json()
    deck_stack = draw_response['cards']

    return deck_stack

def fixFaces(value):
    if value == "ACE":
        return 14
    elif value == "KING":
        return 13
    elif value == "QUEEN":
        return 12
    elif value == "JACK":
        return 11
    else:
        return int(value)

# Need to communicate 3-ACE ranking system with 0-13 ranking system
def convertRank(rank, translation):
    
    # From 0-13 scale to suit scale 
    if translation == "to_suit":
        suit_rank = int(rank) + 2
        
        if suit_rank == 11:
            return "JACK"
        elif suit_rank == 12:
            return "QUEEN"
        elif suit_rank == 13:
            return "KING"
        elif suit_rank == 14:
            return "ACE"
        else:
            return str(suit_rank)

    # From suit scale to 0-13 scale, the way the environment understands it 
    elif translation == "to_env":
        if rank == "JACK":
            return 9
        elif rank == "QUEEN":
            return 10
        elif rank == "KING":
            return 11
        elif rank == "ACE":
            return 12
        else:
            return int(rank) - 2
        

def getCoinFlipCards(deck, coin_flip_result):
    available_cards = deck.copy()

    # 0 means player gets higher card 
    if coin_flip_result == 0:
        high_cards = [card for card in available_cards if fixFaces(card['value']) >= 7]
        player_card = random.choice(high_cards)
        player_value = fixFaces(player_card['value'])

        low_cards = [card for card in available_cards if fixFaces(card['value']) < player_value]
        opponent_card = random.choice(low_cards)

    # When coin flip is 1, opponent goes first and gets higher card 
    else:
        high_cards = [card for card in available_cards if fixFaces(card['value']) >= 7]
        opponent_card = random.choice(high_cards)
        opponent_value = fixFaces(opponent_card['value'])

        low_cards = [card for card in available_cards if fixFaces(card['value']) < opponent_value]
        player_card = random.choice(low_cards)

    return player_card, opponent_card

# Card dealing visual
def deal(deck, coin_flip):
    if not st.session_state.dealing_in_progress:
        # st.session_state.remaining_deck = deck.copy()
        st.session_state.player_hand = []
        st.session_state.opponent_hand = []
        st.session_state.cards_dealt = 0
        st.session_state.dealing_in_progress = True

    if st.session_state.cards_dealt < 14:
        card = st.session_state.deck.pop()

        # Player gets dealt to first 
        if coin_flip == 0:
            if st.session_state.cards_dealt % 2 == 0:
                st.session_state.player_hand.append(card)
            else:
                st.session_state.opponent_hand.append(card)

        # Opponent gets dealt to first 
        else:
            if st.session_state.cards_dealt % 2 == 0:
                st.session_state.opponent_hand.append(card)
            else:
                st.session_state.player_hand.append(card)

        st.session_state.cards_dealt += 1

        if st.session_state.cards_dealt >= 14:
            st.session_state.dealing_complete = True
            st.session_state.env.agent_hand = [convertRank(card['value'], 'to_env') for card in st.session_state.player_hand]
            st.session_state.env.opponent_hand = [convertRank(card['value'], 'to_env') for card in st.session_state.opponent_hand]

    return st.session_state.cards_dealt >= 14

def display_opponent_hand():
    if st.session_state.opponent_hand:
        st.markdown("**Opponent's Hand**")
        
        # Opponent's hand just uses card_back.png

        #1 row total, 1 column per card, max at 7
        cols = st.columns(7)

        # Row spacing for opponent hand
        hand_cards = min(7, len(st.session_state.opponent_hand))
        start_col = (7-hand_cards) // 2

        for each in range(hand_cards):
            with cols[start_col + each]:
                st.image('card_back.png', width=100)

        # Card count
        st.markdown(f'*Cards: {len(st.session_state.opponent_hand)}*', unsafe_allow_html=True)


def display_player_hand():
    if st.session_state.player_hand:
        st.markdown("**Your Hand**")
        # Using 7 cards per row
        cards_per_row = 7
        rows = (len(st.session_state.player_hand) + cards_per_row - 1) // cards_per_row

        for row in range(rows):
            start_idx = row * cards_per_row
            end_idx = min(start_idx + cards_per_row, len(st.session_state.player_hand))
            row_cards = st.session_state.player_hand[start_idx:end_idx]

            # Always create exactly 7 columns for consistency
            cols = st.columns(7)
            
            # Center the cards in this row
            cards_in_row = len(row_cards)
            start_col = (7 - cards_in_row) // 2
            
            for i, card in enumerate(row_cards):
                with cols[start_col + i]:
                    st.image(card['image'], width=100)
                    st.markdown(f'<div style="text-align: center;"><em>{card["value"]}</em></div>', 
                               unsafe_allow_html=True)

        # Card count below, centered
        st.markdown(f'<div style="text-align: center;"><em>Cards: {len(st.session_state.player_hand)}</em></div>', 
                   unsafe_allow_html=True)

def check_and_remove_sets(hand, env_hand, player_name):
    sets_completed = []
    
    # Count cards by rank
    rank_counts = {}
    for card in hand:
        rank = convertRank(card['value'], 'to_env')
        if rank not in rank_counts:
            rank_counts[rank] = []
        rank_counts[rank].append(card)
    
    # Find complete sets (4 cards)
    updated_hand = hand.copy()
    updated_env_hand = env_hand.copy()
    
    for rank, cards in rank_counts.items():
        if len(cards) == 4:
            # Remove all 4 cards from visual hand
            for card in cards:
                if card in updated_hand:
                    updated_hand.remove(card)
            
            # Remove all 4 cards from environment hand
            for _ in range(4):
                if rank in updated_env_hand:
                    updated_env_hand.remove(rank)
            
            # Record the completed set
            display_rank = convertRank(rank, "to_suit")
            sets_completed.append(display_rank)
    
    return updated_hand, updated_env_hand, sets_completed

def check_game_end():
    total_sets = len(st.session_state.get('player_sets', [])) + len(st.session_state.get('opponent_sets', []))
    
    # Game ends when all 13 sets are completed or both hands are empty
    if total_sets >= 13 or (len(st.session_state.player_hand) == 0 and len(st.session_state.opponent_hand) == 0):
      #  return True
        st.session_state.done=True
        return True

    if len(st.session_state.get('player_sets', [])) >= 7 or len(st.session_state.get('opponent_sets', [])) >= 7:
        st.session_state.done=True
        return True
    
    return False

def display_set_completion(sets_completed, player_name):
    if sets_completed:
        for rank in sets_completed:
            st.info(f"{player_name} completed a set of {rank}s! ")
        
        # Add to session state for tracking
        if "player_sets" not in st.session_state:
            st.session_state.player_sets = []
        if "opponent_sets" not in st.session_state:
            st.session_state.opponent_sets = []
            
        if player_name == "You":
            st.session_state.player_sets.extend(sets_completed)
        else:
            st.session_state.opponent_sets.extend(sets_completed)

    

# Start streamlit app with default values
if "validated" not in st.session_state:
    st.session_state.validated = False
if "started" not in st.session_state:
    st.session_state.started = False
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "Easy"
if "coin_flip_shown" not in st.session_state:
    st.session_state.coin_flip_shown = False
if "cards_drawn" not in st.session_state:
    st.session_state.cards_drawn = False
if "player_card" not in st.session_state:
    st.session_state.player_card = None
if "opponent_card" not in st.session_state:
    st.session_state.opponent_card = None
if "game_ready" not in st.session_state:
    st.session_state.game_ready = False
if "dealing_complete" not in st.session_state:
    st.session_state.dealing_complete = False
if "dealing_in_progress" not in st.session_state:
    st.session_state.dealing_in_progress = False
if "player_hand" not in st.session_state:
    st.session_state.player_hand = []
if "opponent_hand" not in st.session_state:
    st.session_state.opponent_hand = []
if "remaining_deck" not in st.session_state:
    st.session_state.remaining_deck = []
if "cards_dealt" not in st.session_state:
    st.session_state.cards_dealt = 0
if "done" not in st.session_state:
    st.session_state.done = False
if "selected_rank" not in st.session_state:
    st.session_state.selected_rank = None
if "player_shown" not in st.session_state:
    st.session_state.player_shown = []


# Pre-game landing screen
if not st.session_state.started:
    # Main greeting text
    st.markdown("<h1 style='text-align: center; '>Go Fish</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; '>Kyle Beach</h5>", unsafe_allow_html=True)
    st.markdown(
    "<h3 style='text-align: center;'>Credit for card images: <a href='https://deckofcardsapi.com' target='_blank'>deckofcardsapi.com</a></h3>",
    unsafe_allow_html=True
    )
    st.markdown(
    "<h3 style='text-align: center;'>Rules: <a href='https://bicyclecards.com/how-to-play/go-fish' target='_blank'>bicyclecards.com</a></h3>",
    unsafe_allow_html=True
    )
    
    # Set up columns for formatting
    col1, col2, col3 = st.columns([1, 1, 1])

    # Dropdown to select difficulty
    with col2:
        st.session_state.difficulty = st.selectbox("Select Difficulty:", options=["Easy", "Medium", "Hard"], index=["Easy", "Medium", "Hard"].index(st.session_state.difficulty))

        # Start game with start button
        if st.button("Start"):
            st.session_state.started = True

            # Load deck, environment, and model depending on difficulty
            with st.spinner("Loading game..."):
                st.session_state.deck = getDeck()
                st.session_state.env = GoFishEnv(mode='play')

                # Set model depending on difficulty
                if st.session_state.difficulty == "Easy":
                    st.session_state.model = PPO.load("GoFish_model") # Change this line for other models
                    st.session_state.env.reset()
                    st.session_state.coin_flip_result = st.session_state.env.coin_flip_result
                    st.session_state.env.set_model(st.session_state.model)

                elif st.session_state.difficulty == "Medium":
                    st.session_state.model = PPO.load("GoFish_model") # Change this line for other models
                    st.session_state.env.reset()
                    st.session_state.coin_flip_result = st.session_state.env.coin_flip_result
                    st.session_state.env.set_model(st.session_state.model)

                elif st.session_state.difficulty == "Hard":
                    st.session_state.model = PPO.load("GoFish_model") # Change this line for other models
                    st.session_state.env.reset()
                    st.session_state.coin_flip_result = st.session_state.env.coin_flip_result
                    st.session_state.env.set_model(st.session_state.model)
            
            st.rerun()


# Start the game 
if st.session_state.started and not st.session_state.game_ready:
    # Set difficulty based on dropdown
    difficulty = st.session_state.difficulty

    # Draw random cards to reflect coin flip results
    if not st.session_state.cards_drawn:
        st.markdown("<h5 style='text-align: center;'>You and your opponent will both draw a card, the higher draw goes first</h5>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Draw Card", key="draw_card_btn", type="primary"):
                st.session_state.player_card, st.session_state.opponent_card = getCoinFlipCards(st.session_state.deck, st.session_state.coin_flip_result)
                st.session_state.cards_drawn = True

                st.rerun()

    # Show drawn cards
    elif st.session_state.cards_drawn and not st.session_state.coin_flip_shown:
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("**Your Card**")
            # Display player card image
            col1.image(st.session_state.player_card['image'], width=200)
            st.markdown(f"*{st.session_state.player_card['value']} of {st.session_state.player_card['suit']}*")
            
        with col2:
            st.write("")  # Empty middle column for spacing
            
        with col3:
            st.markdown("**Opponent Card**")
            # Display player card image
            col3.image(st.session_state.opponent_card['image'], width=200)
            st.markdown(f"*{st.session_state.opponent_card['value']} of {st.session_state.player_card['suit']}*")
            
        # Show who goes first
        if st.session_state.coin_flip_result == 0:
            st.markdown("<h4 style='text-align: center; color: green;'>You drew the higher card! You go first.</h4>", unsafe_allow_html=True)
        else:
            st.markdown("<h4 style='text-align: center; color: red;'>Opponent drew the higher card! Opponent goes first.</h4>", unsafe_allow_html=True)

        # Advance to game button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Begin Game", key="adv_btn", type="primary"):
                st.session_state.coin_flip_shown = True
                st.session_state.game_ready = True
                st.rerun()


    else:
        print("This should be unreachable, line 174")        
                

# Main gameplay  
elif st.session_state.started and st.session_state.game_ready:

    # Begin dealing
    if not st.session_state.dealing_in_progress and not st.session_state.dealing_complete:
        st.session_state.dealing_in_progress = True
        st.rerun()

    # Dealing in progress
    elif st.session_state.dealing_in_progress and not st.session_state.dealing_complete:

        # Opponent hand at top
        display_opponent_hand()

        #st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            st.image('card_back.png', width=100)
            st.markdown(f'*Deck Cards: {len(st.session_state.deck)}*')

        #st.markdown("---")

        display_player_hand()

        if st.session_state.cards_dealt < 14:
            time.sleep(0.2)
            deal(st.session_state.deck, st.session_state.coin_flip_result)
            st.rerun()

    # Dealing is done, cards are on display and game can start 
    elif st.session_state.dealing_complete:
        # Opponent hand in top row
        display_opponent_hand()

        # keep track of sets
        if st.session_state.get('player_sets') or st.session_state.get('opponent_sets'):
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                player_score = len(st.session_state.get('player_sets', []))
                st.markdown(f"**Your Sets: {player_score}**")
            with col3:
                opponent_score = len(st.session_state.get('opponent_sets', []))
                st.markdown(f"**Opponent Sets: {opponent_score}**")

        # spacing
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            st.image('card_back.png', width=100)
            st.markdown(f'*Deck Cards: {len(st.session_state.deck)}*')

        st.markdown("---")

        # Player hand at the bottom
        display_player_hand()
    

        # Actual game starts 
        base_env = st.session_state.env
        env = FlattenObservation(base_env)

        if st.session_state.done == False:
            # Ensure hands are playable 
            base_env._check_empty_hand()

            # Player's turn
            if base_env.agent_turn:
                rank_counts = {}

                for card in st.session_state.player_hand:
                    rank = convertRank(card['value'], 'to_env')
                    rank_counts[rank] = rank_counts.get(rank, 0) + 1
                    
                player_shown = [rank for rank, count in rank_counts.items() if count > 0]

                st.session_state.player_shown = player_shown

                # Show legal moves if player has cards
                if player_shown:
                    st.markdown("**Choose a rank to ask for:**")

                    cols = st.columns(len(player_shown))

                    for idx, rank in enumerate(player_shown):
                        with cols[idx]:
                            # Show the number of each rank
                            # count = base_env.agent_hand.count(rank)
                            # count = st.session_state.player_hand.count(rank)

                            count = 0
                            for card in st.session_state.player_hand:
                                card_rank = convertRank(card['value'], "to_env")
                                if card_rank == rank:
                                    count += 1

                            display = convertRank(rank, "to_suit")
                            display = str(display) if isinstance(display, int) else display
                                    
                            btn_lbl = f'{display}\n({count})'

                            if st.button(btn_lbl, key=f'rank_{rank}',
                                         type = "primary" if st.session_state.selected_rank == rank else "secondary"):
                                st.session_state.selected_rank = rank

                    if st.session_state.selected_rank is not None:
                        display = convertRank(st.session_state.selected_rank, "to_suit")
                        display = str(display) if isinstance(display, int) else display
                        st.markdown(f"**Selected:** Rank {display}")
            
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col2:
                            if st.button("Confirm", key="confirm", type="primary"):
                                # Translate action from suit to env
                                action = st.session_state.selected_rank
                                # translated_action = convertRank(action, "to_env")
                                translated_action = action

                                # Execute move
                                base_env.agent_hand = [convertRank(card['value'], 'to_env') for card in st.session_state.player_hand]
                                base_env.opponent_hand = [convertRank(card['value'], 'to_env') for card in st.session_state.opponent_hand]
                                obs, reward, done, _, info = env.step(translated_action)

                                # See if ask was successful
                                success = info.get("agent_success", False)

                                if success:
                                    moved_cards = []
                                    remaining_opponent_hand = []
                                    for card in st.session_state.opponent_hand:
                                        card_rank_val = convertRank(card['value'], "to_env")

                                        # Translated action is the requested rank
                                        # Successful ask
                                        if card_rank_val == translated_action:
                                            moved_cards.append(card)

                                        # Unsuccessful ask
                                        else:
                                            remaining_opponent_hand.append(card)

                                    st.session_state.opponent_hand = remaining_opponent_hand
                                    st.session_state.player_hand.extend(moved_cards)

                                    # see completed sets
                                    st.session_state.player_hand, updated_agent_hand, player_sets = check_and_remove_sets(st.session_state.player_hand, [convertRank(card['value'], 'to_env') for card in st.session_state.player_hand],"You")

                                    display_set_completion(player_sets, "You")

                                    
                                    base_env.agent_hand = updated_agent_hand
                                    base_env.opponent_hand = [convertRank(card['value'], 'to_env') for card in st.session_state.opponent_hand]
                                        
                                else:
                                   if st.session_state.deck:
                                       draw = st.session_state.deck.pop()
                                       st.session_state.player_hand.append(draw)

                                       base_env.agent_hand.append(convertRank(draw['value'], 'to_env'))

                                       st.session_state.player_hand, updated_agent_hand, player_sets = check_and_remove_sets(st.session_state.player_hand,[convertRank(card['value'], 'to_env') for card in st.session_state.player_hand],"You")


                                       # display_set_completion(player_sets, "You")
                                       if check_game_end():
                                           st.rerun()

                                # Reset selection
                                st.session_state.selected_rank = None

                                # Update game state
                                st.session_state.done = done

                                if success:
                                    st.info(f"Successful ask! You took all opponent's {display}s and go again.")
                                    #display_set_completion(player_sets, "You")
                                    #st.session_state.player_hand, updated_agent_hand, player_sets = check_and_remove_sets(st.session_state.player_hand, [convertRank(card['value'], 'to_env') for card in st.session_state.player_hand],"You")
                                    if st.button("Continue", key="cont_btn_after_successful_ask"):
                                        if check_game_end():
                                            st.rerun()
                                        st.rerun()
                                else:
                                    st.info(f"No {display}s. You drew a card. Opponent's turn.")
                                    #display_set_completion(player_sets, "You")
                                    #st.session_state.player_hand, updated_agent_hand, player_sets = check_and_remove_sets(st.session_state.player_hand, [convertRank(card['value'], 'to_env') for card in st.session_state.player_hand],"You")
                                    if st.button("Continue", key="cont_btn_after_failed_ask"):
                                        if check_game_end():
                                            st.rerun()
                                        st.rerun()

                    else:
                        st.info("Select a rank to ask for.")

                else:
                    if check_game_end():
                        st.rerun()

            # Opponent turn
            else:

                base_env.agent_hand = [convertRank(card['value'], 'to_env') for card in st.session_state.player_hand]
                base_env.opponent_hand = [convertRank(card['value'], 'to_env') for card in st.session_state.opponent_hand]
                obs, reward, done, _, info = env.step(0)
                st.session_state.done = done

                action = info.get("opponent_action", None)
                success = info.get("opponent_success", False)

                if action is not None:
                    translated_action = convertRank(action, "to_suit")

                    if success:
                        moved_cards = []
                        remaining_player_hand = []

                        # Compare env version of card to each suit card in player's hand
                        for card in st.session_state.player_hand:
                            card_rank_val = convertRank(card['value'], 'to_env')
                            if card_rank_val == action:
                                moved_cards.append(card)
                            else:
                                remaining_player_hand.append(card)

                        st.session_state.player_hand = remaining_player_hand
                        st.session_state.opponent_hand.extend(moved_cards)

                        # Check for completed sets after opponent gets cards
                        st.session_state.opponent_hand, updated_opponent_hand, opponent_sets = check_and_remove_sets(st.session_state.opponent_hand, [convertRank(card['value'], 'to_env') for card in st.session_state.opponent_hand], "Opponent")

                        base_env.agent_hand = [convertRank(card['value'], 'to_env') for card in st.session_state.player_hand]
                        base_env.opponent_hand = updated_opponent_hand

                        # Display set completion notification                     
                        st.info(f"Opponent successfully took your {translated_action}s")
                        display_set_completion(opponent_sets, "Opponent")
                        
                        if st.button("Continue", key="cont_btn_after_info"):
                            if check_game_end():
                                st.rerun()
                    else:
                        # go fish if fail
                        if st.session_state.deck:
                            draw = st.session_state.deck.pop()
                            st.session_state.opponent_hand.append(draw)

                            # Check for completed sets after opponent draws
                            st.session_state.opponent_hand, updated_opponent_hand, opponent_sets = check_and_remove_sets(st.session_state.opponent_hand, [convertRank(card['value'], 'to_env') for card in st.session_state.opponent_hand], "Opponent")
        
                            base_env.opponent_hand = updated_opponent_hand
        
                            # Display set completion notification
                            
                            st.info(f"Opponent asked for {translated_action}s, but you had none. Opponent drew a card.")
                            display_set_completion(opponent_sets, "Opponent")
                            st.session_state.opponent_hand, updated_opponent_hand, opponent_sets = check_and_remove_sets(st.session_state.opponent_hand, [convertRank(card['value'], 'to_env') for card in st.session_state.opponent_hand], "Opponent")
                            if st.button("Continue", key="cont_btn_after_oppo"):
                                if check_game_end():
                                    st.rerun()
                                st.rerun()

                        else:
                            st.info(f"Opponent asked for {translated_action}s, but you had none. No more cards to draw from.")
                            display_set_completion(opponent_sets, "Opponent")
                            st.session_state.opponent_hand, updated_opponent_hand, opponent_sets = check_and_remove_sets(st.session_state.opponent_hand, [convertRank(card['value'], 'to_env') for card in st.session_state.opponent_hand], "Opponent")
                            if st.button("Continue", key="cont_btn_after_info2"):
                                if check_game_end():
                                    st.rerun()
                                st.rerun()
                        

                else:
                    st.warning("Opponent had no legal move")

        if st.session_state.done == True:

            st.markdown("Game Over")

            player_score = len(st.session_state.get('player_sets', []))
            opponent_score = len(st.session_state.get('opponent_sets', []))
    
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Your Sets:** {player_score}")
                if st.session_state.get('player_sets'):
                    st.write(", ".join(str(s) for s in st.session_state.player_sets))
    
            with col2:
                st.markdown(f"**Opponent Sets:** {opponent_score}")
                if st.session_state.get('opponent_sets'):
                    st.write(", ".join(str(s) for s in st.session_state.opponent_sets))
    
            if player_score > opponent_score:
                st.success("You Win!")
            elif opponent_score > player_score:
                st.error("Opponent Wins!")
            else:
                st.info("It's a Tie!")

            if st.button("Play Again"):
                # Reset all game state
                for key in list(st.session_state.keys()):
                    if key != 'started':
                        del st.session_state[key]
                    st.session_state.started = False
                    st.rerun()

        
        #st.rerun()
