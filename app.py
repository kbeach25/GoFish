import streamlit as st
import requests
import random
from PIL import Image
from io import BytesIO

# Start streamlit app with default values
if "validated" not in st.session_state:
    st.session_state.validated = False
if "started" not in st.session_state:
    st.session_state.started = False
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "Easy"

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
            st.rerun()
    
