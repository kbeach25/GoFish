import streamlit as st
import requests
import random
from PIL import Image
from io import BytesIO
import ai_easy

def getDeck(players):
    # 2 players only use one deck
    if players == 2:
        # Get deck from deckofcardsapi
        url = "https://deckofcardsapi.com/api/deck/new/shuffle/?deck_count=1"
        response = requests.get(url).json()
        id = response['deck_id']

        # put shuffled cards into deck as a stack
        draw_url = f"https://deckofcardsapi.com/api/deck/{id}/draw/?count=52"
        draw_response = requests.get(draw_url).json()
        cards = draw_response['cards']

        deck_stack = cards[::-1] # Reverse order so first card drawn is on the bottom

    # 3+ players use two decks
    else:
        # Get deck from deckofcardsapi
        url = "https://deckofcardsapi.com/api/deck/new/shuffle/?deck_count=2"
        response = requests.get(url).json()
        id = response['deck_id']

        # put shuffled cards into deck as a stack
        draw_url = f"https://deckofcardsapi.com/api/deck/{id}/draw/?count=104"
        draw_response = requests.get(draw_url).json()
        cards = draw_response['cards']

        deck_stack = cards[::-1] # Reverse order so first card drawn is on the bottom
        
    return deck_stack

# Only need the first card
def deal():
    if len(st.session_state.deck) == 0:
        # Subject to change
        st.warning("Deck is empy")
        return None
    
    return st.session_state.deck.pop()

# Get value of each card for sorting and comparison
def cardValue(card):
    order = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7,
             '8':8, '9':9, '10':10, 'JACK':11, 'QUEEN':12, 'KING':13, 'ACE':14}
    return order[card['value']]

# Players choose three cards to start with, the other three are kept in the second hand
def playerSecondHand(second_hand_cards):
    st.markdown("<h3>Choose 3 cards for starting hand, other 3 will be kept in second hand<h3/>", unsafe_allow_html=True)
    scroll_style = """
    <style>
    .scrollable-container {
        max-height: 200px;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 10px;
        background-color: #f9f9f9;
    }
    </style>
"""
    st.markdown(scroll_style, unsafe_allow_html=True)
    sec_hand_6_container = st.container()

    with sec_hand_6_container:
        st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)

        # Display cards in rows of 3, make them clickable
        for i in range(0, len(second_hand_cards), 3):
            cols = st.columns(3)
            
            for j, card in enumerate(second_hand_cards[i:i+3]):
                card_idx = i + j
                img_col = cols[j]
                img_col.image(card['image'], width=200)

                selected = card_idx in st.session_state.selections
                btn_label = f"{'Deselect' if selected else 'Select'} {card['value']} of {card['suit']}"
    
                if img_col.button(btn_label, key=f"card_btn_{card_idx}"):
                    if selected:
                        st.session_state.selections.remove(card_idx)
                    else:
                        if len(st.session_state.selections) >= 3:
                            st.session_state.selections.pop(0)
                        st.session_state.selections.append(card_idx)
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Confirm Starting Hand"):
        if len(st.session_state.selections) != 3:
            st.warning("You must choose 3 cards")
        else:
            sel_idx = list(st.session_state.selections)
            st.session_state.current_hand = [second_hand_cards[i] for i in sel_idx]
            st.session_state.second_hand = [card for idx, card in enumerate(second_hand_cards) if idx not in sel_idx]
            st.session_state.first_hand = True
            st.rerun()

            
    
    

# start streamlit app
if "validated" not in st.session_state:
    st.session_state.validated = False
if "code" not in st.session_state:
    st.session_state.code = ""
if "started" not in st.session_state:
    st.session_state.started = False
if "first_hand" not in st.session_state:
    st.session_state.first_hand = False
if "selections" not in st.session_state:
    st.session_state.selections = []
if "current_hand" not in st.session_state:
    st.session_state.current_hand = []
if "second_hand" not in st.session_state:
    st.session_state.second_hand = []
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "Easy"
if "num_players" not in st.session_state:
    st.session_state.num_players = 2
        

# This is what shows up before you started the game 
if not st.session_state.started:
    # Markdown used for more text customization
    st.markdown("<h1 style='text-align: center;'>Palace Card Game</h1>", unsafe_allow_html=True)
    st.markdown(
    "<h3 style='text-align: center;'>Credit for card images: <a href='https://deckofcardsapi.com' target='_blank'>deckofcardsapi.com</a></h3>",
    unsafe_allow_html=True
)
    st.markdown('''
    Rules: Different than you'll see online, but these are the rules I was taught
    and have always played by.

    2 players use one deck, 3-5 players use two. 

    Dealer deals a card from shuffed deck to each player until each player has three.
    cards are to remain face down and cannot be seen until all of their other cards
    are gone. These make up a player's "back row".

    Then, dealer deals a card from shuffled deck to each player until they have six.
    Players choose three cards to start with and three cards to keep face down, their
    "second hand".

    Whichever player has the lowest card goes first. If it's a tie, a coin flip or
    dice roll is used to determine who goes first. Rotation is other wise clockwise.

    RULES ARE INCOMPLETE
    '''
                )


    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        # Dropdowns to choose players and difficulty
        st.session_state.difficulty = st.selectbox("Select Difficulty:", options=["Easy", "Medium", "Hard"], index=["Easy", "Medium", "Hard"].index(st.session_state.difficulty))
        st.session_state.num_players = st.selectbox("Number of Players:", options=[2, 3, 4, 5], index=[2, 3, 4, 5].index(st.session_state.num_players))

        # Run game with go button
        if st.button("Go"):
            st.session_state.started = True
            st.session_state.first_hand = False
            st.session_state.selections = []
            st.session_state.current_hand = []
            st.session_state.second_hand = []
            st.session_state.deck = getDeck(st.session_state.num_players)
            st.session_state.px = [[[], [], []] for _ in range(st.session_state.num_players)]
            st.session_state.dealt = False
            st.rerun()

# This is what shows up once the game has started
if st.session_state.started:
    players = st.session_state.num_players
    dif = st.session_state.difficulty

    # Need deck and hand initialization
    if 'deck' not in st.session_state:
        st.session_state.deck = getDeck(players)
    if 'px' not in st.session_state:
        st.session_state.px = [[[], [], []] for _ in range(players)]
    if 'dealt' not in st.session_state:
        st.session_state.dealt = False

    deck_view = st.empty()
    with deck_view.container():
        deck_col = st.columns([1, 2, 1])
        with deck_col[1]:
            deck_count = st.empty()
            st.image("card_back.png", width=200)
            deck_count.markdown(f'### Deck Cards: {len(st.session_state.deck)}')

    if not st.session_state.dealt:
        # Assign random player to begin dealing
        starting_idx = random.randint(0, players-1)
        st.markdown(f"First player to get dealt a card: P{starting_idx + 1}") 

        # Back row dealing
        back_row_cards = players * 3
    
        for each in range(back_row_cards):
            idx = each % players # loop through players
            card = deal()
            st.session_state.px[idx][2].append(card)
            deck_view.markdown(f'### Deck: {len(st.session_state.deck)}')


        # Second hand dealing
        second_hand_cards = players * 6
        for each in range(second_hand_cards):
            idx = each % players # loop through players
            card = deal()
            st.session_state.px[idx][1].append(card)
            deck_view.markdown(f'### Deck Cards: {len(st.session_state.deck)}')

        # End dealing process to avoid re-dealing with every rerun
        st.session_state.dealt = True


    # P1 must see hand to choose cards, px[0] is human player
    p1_second_hand_6 = sorted(st.session_state.px[0][1], key=cardValue)

    # Second hand selection for player
    if not st.session_state.first_hand:
        if "p1_sec_hand_cache" not in st.session_state:
            st.session_state.p1_sec_hand_cache = p1_second_hand_6
        playerSecondHand(st.session_state.p1_sec_hand_cache)
    else:
        # Starting hand is present and face up
        st.markdown("### Starting Hand:")
        current_hand = st.session_state.current_hand

        cols = st.columns(3)
        for i, card in enumerate(current_hand):
            with cols[i]:
                st.image(card['image'], width=200)

        # Second hand is present and face down
        sec_hand = st.session_state.second_hand
        st.markdown(f'### Second Hand: {len(sec_hand)}')

        cols = st.columns(3)
        for i, card in enumerate(sec_hand):
            with cols[i]:
                st.image("card_back.png", width=200)

        # Back row is present and face down
        st.markdown("### Back Row: 3")

        cols = st.columns(3)
        for i in range(3):
            with cols[i]:
                st.image("card_back.png", width=200)

        # CPU second hand selector
        for idx, player in enumerate(st.session_state.px):
            if idx == 0:
                # Human player, skip AI logic
                continue
            else:
                ### THIS IS TEMPORARY
                cpu_hand = player[1]  # second hand of 6 cards
                
                for card in cpu_hand:
                    print(f'{card["value"]} of {card["suit"]}')

                ####################
                ai_easy.second_hand_easy(player)

                ### MORE TEMPORARY
                print("\nStarting Hand:")
                for card in player[0]:
                   print(f'{card["value"]} of {card["suit"]}')

                print("\nSecond Hand")
                for card in player[1]:
                   print(f'{card["value"]} of {card["suit"]}')
                print("\n")
