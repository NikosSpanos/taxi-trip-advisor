import streamlit as st
import random
import os
import configparser
import sys
import mlflow
import numpy as np
import datetime
import re
sys.path.append('./src')
from chatbot_modules import stream_simulation
from ml_modules import load_label_encoder


def ai_advisor():
    st.set_page_config(page_title="AI advisor")
    st.title("Your AI search engine for taxi tips!")

    # Initialize ml artifacts
    config = configparser.ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    config.read(os.path.join(parent_dir, "config", "config.ini"))
    application_path:str = config.get("settings", "application_path")
    artifact_path:str = os.path.join(application_path, "model_artifacts",)
    artifact_dt:str = config.get("ai-advisor", "artifacts_dt")
    pu_encoder_name:str = config.get("ai-advisor", "pickup_loc_encoder")
    do_encoder_name:str = config.get("ai-advisor", "dropoff_loc_encoder")
    pu_encoder_path:str = os.path.join(artifact_path, artifact_dt, pu_encoder_name)
    do_encoder_path:str = os.path.join(artifact_path, artifact_dt, do_encoder_name)
    duration_predictor:str = config.get("ai-advisor", "duration_model")
    cost_predictor:str = config.get("ai-advisor", "cost_model")

    duration_model_uri = os.path.join(application_path, "mlruns", artifact_dt, duration_predictor)
    duration_model_uri = "file://{0}".format(duration_model_uri)

    cost_model_uri = os.path.join(application_path, "mlruns", artifact_dt, cost_predictor)
    cost_model_uri = "file://{0}".format(cost_model_uri)
    # Streamlit emojis: https://emojidb.org/streamlit-emojis

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    triggering_appriciation_words = ["Thanks", "Thank you", "You are the best", "helpful"]
    triggering_advice_words = ["from", "to", "at"]
    
    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if all(word.lower() in str.lower(st.session_state.messages[-1]["content"]) for word in triggering_advice_words):
            print("1. Where am I?")

            print("2. sentence:", prompt)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Extract pickup and dropoff locations
            # tokens = prompt.split(" ")
            # print(tokens)
            # pickup_index = tokens.index("from") + 1
            # dropoff_index = pickup_index + 2

            pickup_match = re.search(r'from\s+(.*?)\s+to', prompt, re.IGNORECASE)
            dropoff_match = re.search(r'(?:.*\bto\s)(.*?)(?=\s*at\b)', prompt, re.IGNORECASE)

            # Extract pickup time
            # pickup_time_expr = re.search(r'(\d+):', prompt)
            pickup_time_expr = prompt.split(":")[0]
            pickup_time = pickup_time_expr.split(" ")[-1]

            # print(f"From index: {pickup_index}")
            # print(f"To index: {dropoff_index}")
            print(f"From match: {pickup_match}")
            print(f"To match: {dropoff_match}")

            # if pickup_index and dropoff_index:
            if pickup_match and dropoff_match:
                if pickup_time.isdigit():
                    # pickup_location = tokens[pickup_index]
                    # dropoff_location = tokens[dropoff_index]
                    pickup_location = pickup_match.group(1).strip()
                    dropoff_location = dropoff_match.group(1).strip()
                    print(f"From: {pickup_location}")
                    print(f"To: {dropoff_location}")
                    print(f"At: {pickup_time}")

                    #Load label encoder for pickup, dropoff locations
                    pu_encoder = load_label_encoder(pu_encoder_path)
                    do_encoder = load_label_encoder(do_encoder_path)
                    pickup_time = int(pickup_time)
                    # print(pu_encoder.classes_)
                    # print(do_encoder.classes_)
                    # if (pickup_location not in pu_encoder.classes_) or (dropoff_location not in do_encoder.classes_):
                    if (pickup_location not in pu_encoder.classes_) and (dropoff_location in do_encoder.classes_):
                        assistant_response = f"❌Sorry but I don't have information about {pickup_location}. Please try another pickup point."
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                        with st.chat_message("assistant"):
                            mssg_placeholder, assistant_response = stream_simulation(assistant_response)
                        mssg_placeholder.markdown(assistant_response)
                    elif (pickup_location in pu_encoder.classes_) and (dropoff_location not in do_encoder.classes_):
                        assistant_response = f"❌Sorry but I don't have information about {dropoff_location}. Please try another dropoff/destination point."
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                        with st.chat_message("assistant"):
                            mssg_placeholder, assistant_response = stream_simulation(assistant_response)
                        mssg_placeholder.markdown(assistant_response)
                    elif (pickup_location not in pu_encoder.classes_) and (dropoff_location not in do_encoder.classes_):
                        assistant_response = f"❌Sorry but I don't have information about {pickup_location} and {dropoff_location}. Please try another combination."
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                        with st.chat_message("assistant"):
                            mssg_placeholder, assistant_response = stream_simulation(assistant_response)
                        mssg_placeholder.markdown(assistant_response)
                    else:
                        loaded_duration_model = mlflow.pyfunc.load_model(duration_model_uri)
                        loaded_cost_model = mlflow.pyfunc.load_model(cost_model_uri)
                        pu_index = pu_encoder.transform(np.array([pickup_location]))[0]
                        do_index = do_encoder.transform(np.array([dropoff_location]))[0]
                        predicted_duration = loaded_duration_model.predict(np.array([[pu_index, do_index, pickup_time]]))[0]
                        predicted_cost = loaded_cost_model.predict(np.array([[pu_index, do_index, pickup_time, predicted_duration]]))[0]
                        seconds = int(predicted_duration * 60)
                        delta = datetime.timedelta(seconds=seconds)
                        if ( (pickup_time in range(7,11)) or (pickup_time in range(16,20)) ) or (pickup_time in [20,21,22,23,0,1,2,3,4,5,6]):
                            extra = 1.0
                        else:
                            extra = 0.50
                        mta_tax = 0.50
                        improvement_surcharge = 0.30
                        predicted_cost = predicted_cost + mta_tax + extra + improvement_surcharge
                        assistant_response = f"🚕 Your trip will have an average duration of {str(delta)}⏱ and it will cost approximately ${np.round(predicted_cost, 2)}💸 (including taxes)."
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                        with st.chat_message("assistant"):
                            mssg_placeholder, assistant_response = stream_simulation(assistant_response)
                        mssg_placeholder.markdown(assistant_response)
                else:
                    assistant_response = "❌Attention! You didn't type your **pickup** time. Annotate *<TIME>* by using the keyword :blue[at] (24h scale)."
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    with st.chat_message("assistant"):
                        mssg_placeholder, assistant_response = stream_simulation(assistant_response)
                    mssg_placeholder.markdown(assistant_response)
            else:
                assistant_response = "❌Attention! You didn't type your **pickup** and **dropoff** locations. Annotate *<LOCATION>* by using the keywords :blue[from] and :blue[to]."
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                with st.chat_message("assistant"):
                    mssg_placeholder, assistant_response = stream_simulation(assistant_response)
                mssg_placeholder.markdown(assistant_response)

        elif len(st.session_state.messages) == 1:
            # Provide assistant response
            assistant_response = random.choice([
                "👋 Hello, I am your personal AI advisor for taxi trip's cost and duration. How can I help you?",
                "👋 Hi, there! Any taxi trip's cost and duration advice for today?",
                "👋 Greetings from your personal AI advisor. How can I help you?",
            ])
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            # Simulate stream
            with st.chat_message("assistant"):
                mssg_placeholder, assistant_response = stream_simulation(assistant_response)
            mssg_placeholder.markdown(assistant_response)
        elif any(word.lower() in str.lower(st.session_state.messages[-1]["content"]) for word in triggering_appriciation_words):
            assistant_response = "😎You're welcome. Have a great trip and keep safe!"
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                mssg_placeholder, assistant_response = stream_simulation(assistant_response)
            mssg_placeholder.markdown(assistant_response)
        else:
            # Provide assistant response
            assistant_response = "❗Sorry, I am an ai taxi-trip advisor. Please write something like 'I would like to go :blue[from] <PLACE> :blue[to] <PLACE> :blue[at] <PICKUP TIME>' "
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            # Simulate stream
            with st.chat_message("assistant"):
                mssg_placeholder, assistant_response = stream_simulation(assistant_response)
            mssg_placeholder.markdown(assistant_response)
if __name__ == "__main__":
    ai_advisor()