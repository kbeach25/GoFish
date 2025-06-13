# Easy Difficulty module
import random

# Second hand selection, needs 6 cards for this
def second_hand_easy(player):
    cards = player[1]

    # for easy mode, second hand selection is random
    starting_hand = random.sample(cards, 3)

    remaining_sh = [card for card in cards if card not in starting_hand]

    player[0] = starting_hand
    player[1] = remaining_sh

