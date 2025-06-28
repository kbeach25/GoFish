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
                    st.session_state.model = PPO.load("GoFish_model")
                    st.session_state.env.reset()
                    st.session_state.coin_flip_result = st.session_state.env.coin_flip_result
                    st.session_state.env.set_model(st.session_state.model)
            
            st.rerun()


# Start the game 
if st.session_state.started:
    # Set difficulty based on dropdown
    difficulty = st.session_state.difficulty

    # Draw random cards to reflect coin flip results
    if not st.session_state.coin_flip_shown and not st.session_state.cards_drawn:
        st.markdown("<h5 style='text-align: center;'>You and your opponent will both draw a card, the higher draw goes first</h5>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Draw Card", key="draw_card_btn", type="primary"):
                st.session_state.coin_flip_shown = True
                st.session_state.cards_drawn = True

                st.session_state.player_card, st.session_state.opponent_card = getCoinFlipCards(st.session_state.deck, st.session_state.coin_flip_result)

                # time.sleep(3)
                st.rerun()

        if st.session_state.get('cards_drawn', False):
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.markdown("**Player Card**")
                # Display player card image
                col1.image(st.session_state.player_card['image'], width=200)
                st.markdown(f"*{st.session_state.player_card['value']} of {st.session_state.player_card['suit']}*")
                print("success 1")
            
            with col2:
                st.write("")  # Empty middle column for spacing
            
            with col3:
                st.markdown("**Opponent Card**")
                # Display player card image
                col3.image(st.session_state.opponent_card['image'], width=200)
                st.markdown(f"*{st.session_state.opponent_card['value']} of {st.session_state.player_card['suit']}*")

            time.sleep(3)
            
            # Show who goes first
            if st.session_state.coin_flip_result == 0:
                st.markdown("<h4 style='text-align: center; color: green;'>You drew the higher card! You go first.</h4>", unsafe_allow_html=True)
            else:
                st.markdown("<h4 style='text-align: center; color: red;'>Opponent drew the higher card! Opponent goes first.</h4>", unsafe_allow_html=True)
                

    else: 
        # Set up "deck" like you would see on a table
        deck_view = st.empty()

        # Use column formatting again
        with deck_view.container():

            # Temporary spacing solution
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                st.image('card_back.png', width=100)
                st.markdown(f'*Deck Cards: {len(st.session_state.deck)}*')

            coin_flip_result = st.session_state.coin_flip_result
            st.markdown(coin_flip_result)
        



        

    
    
