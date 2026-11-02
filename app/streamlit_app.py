import os
import requests

import streamlit as st


api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")


st.title("Spaceship Titanic Predictor")
st.write(f"API URL: {api_base_url}")


try:
    response = requests.get(f"{api_base_url}/health", timeout=5)
    response.raise_for_status()
    health_data = response.json()

    if health_data.get("status") == "ok":
        st.success("The prediction API is available.")
    else:
        st.error("The prediction API is currently unavailable.")
except (requests.exceptions.RequestException, ValueError):
    st.error("The prediction API is currently unavailable.")

passenger_id = st.text_input(
    "Passenger ID",
    value="0001_01",
    help="Required format: ####_##, for example 0001_01"
)

home_planet = st.selectbox(
    "Home planet",
    options=["Earth", "Europa", "Mars"],
    help="Select the passenger's home planet."
)

destination = st.selectbox(
    "Destination",
    options=["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"],
    help="Select the passenger's destination"
)

cryo_sleep = st.checkbox(
    "Passenger was in cryosleep",
    help="Select this if the passenger was placed in suspended animation."
)

st.write("Cabin")

cabin_deck = st.selectbox(
    "Deck",
    options=["A", "B", "C", "D", "E", "F", "G", "T"]
)

cabin_number = st.number_input(
    "Cabin number",
    min_value=0,
    max_value=1894,
    value=0,
    step=1
)

cabin_side = st.selectbox(
    "Cabin side",
    options=["P", "S"],
    help="P means Port; S means Starboard."
)

cabin = f"{cabin_deck}/{cabin_number}/{cabin_side}"

age = st.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=30,
    step=1,
    help="Enter the passenger's age in whole years."
)

vip = st.checkbox(
    "VIP passenger",
    help="Select this if the passenger had VIP status."
)

room_service = st.number_input(
    "Room service spending",
    min_value=0.0,
    value=0.0,
    step=1.0,
    help="Enter the passenger's room service charges."
)

food_court = st.number_input(
    "Food court spending",
    min_value=0.0,
    value=0.0,
    step=1.0,
    help="Enter the passenger's food court charges."
)

shopping_mall = st.number_input(
    "Shopping mall spending",
    min_value=0.0,
    value=0.0,
    step=1.0,
    help="Enter the passenger's shopping mall charges."
)

spa = st.number_input(
    "Spa spending",
    min_value=0.0,
    value=0.0,
    step=1.0,
    help="Enter the passenger's spa charges."
)

vr_deck = st.number_input(
    "VR deck spending",
    min_value=0.0,
    value=0.0,
    step=1.0,
    help="Enter the passenger's VR deck charges.",
)

predict_button = st.button(
    "Predict transportation",
    type="primary"
)

if predict_button:
    payload = {
        "PassengerId": passenger_id,
        "HomePlanet": home_planet,
        "CryoSleep": cryo_sleep,
        "Cabin": cabin,
        "Destination": destination,
        "Age": age,
        "VIP": vip,
        "RoomService": room_service,
        "FoodCourt": food_court,
        "ShoppingMall": shopping_mall,
        "Spa": spa,
        "VRDeck": vr_deck
    }

    try:
        prediction_response = requests.post(
            f"{api_base_url}/predict",
            json=payload,
            timeout=10
        )

        if prediction_response.status_code == 422:
            st.error(
                "Some passenger information is invalid. "
                "Check the Passenger ID and cabin formats."
            )
        else:
            prediction_response.raise_for_status()
            prediction_data = prediction_response.json()

            transported = prediction_data["transported"]
            probability = prediction_data["probability"]

            if transported:
                st.success("Prediction: Transported")
            else:
                st.info("Prediction: Not transported")

            st.metric(
                "Estimated probability of being transported",
                f"{probability:.1%}"
            )

            st.caption(
                "This probability is a model estimate, not a certainty."
            )

    except requests.exceptions.Timeout:
        st.error("The prediction API took too long to respond.")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the prediction API.")
    except (requests.exceptions.RequestException, ValueError):
        st.error("The prediction could not be completed.")