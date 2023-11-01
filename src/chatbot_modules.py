import time
import streamlit as st

def stream_simulation(assistant_response):
    message_placeholder = st.empty()
    full_response = ""
    for chunk in assistant_response.split():
        full_response += chunk + " "
        time.sleep(0.04)
        message_placeholder.markdown(full_response + "▌")
    return (message_placeholder, full_response)