import os
import random
import time
from html import escape

import requests
import streamlit as st


api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Spaceship Titanic · Boarding Terminal",
    page_icon="🚀",
    layout="centered",
)


SPACE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');

:root {
    --glow: #7cf7ff;
    --gold: #ffd75e;
    --void: #b47cff;
    --ink: #d9e6ff;
}

.stApp {
    background:
        radial-gradient(1px 1px at 10% 20%, #fff, transparent),
        radial-gradient(1px 1px at 30% 80%, #fff, transparent),
        radial-gradient(1.5px 1.5px at 50% 40%, #cfe8ff, transparent),
        radial-gradient(1px 1px at 70% 10%, #fff, transparent),
        radial-gradient(1px 1px at 85% 65%, #fff, transparent),
        radial-gradient(1.5px 1.5px at 15% 55%, #ffe9c4, transparent),
        radial-gradient(1px 1px at 60% 90%, #fff, transparent),
        radial-gradient(1px 1px at 92% 30%, #cfe8ff, transparent),
        radial-gradient(ellipse at 80% 0%, rgba(120, 60, 200, 0.35), transparent 60%),
        radial-gradient(ellipse at 0% 100%, rgba(30, 90, 180, 0.35), transparent 55%),
        #04050f;
    background-size: 400px 400px, 300px 300px, 500px 500px, 350px 350px,
                     450px 450px, 600px 600px, 380px 380px, 420px 420px,
                     100% 100%, 100% 100%, 100% 100%;
    animation: drift 120s linear infinite;
    color: var(--ink);
}

@keyframes drift {
    from { background-position: 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0; }
    to   { background-position: -400px 200px, -300px 150px, -500px 250px, -350px 175px,
                                -450px 225px, -600px 300px, -380px 190px, -420px 210px,
                                0 0, 0 0, 0 0; }
}

[data-testid="stHeader"] { background: transparent; }

h1, h2, h3, p, label, input, button, .stMarkdown, .mono,
.hero *, .crawl, .crawl *, .verdict, .verdict *, .footnote {
    font-family: 'VT323', monospace !important;
    letter-spacing: 0.04em;
}

p, label, input, .stMarkdown { font-size: 1.2rem; }

.hero {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.hero h1 {
    font-size: 3.4rem;
    color: var(--gold);
    text-shadow: 0 0 18px rgba(255, 215, 94, 0.55);
    margin-bottom: 0;
}
.hero .sub {
    color: var(--glow);
    letter-spacing: 0.35em;
    font-size: 1.1rem;
}

.crawl {
    border-left: 3px solid var(--gold);
    background: rgba(10, 14, 35, 0.7);
    padding: 1rem 1.25rem;
    margin: 1.25rem 0;
    line-height: 1.5;
    font-size: 1.3rem;
    color: #e8eaff;
}

.panel-title {
    font-family: 'VT323', monospace;
    color: var(--glow);
    font-size: 1.2rem;
    letter-spacing: 0.25em;
    border-bottom: 1px solid rgba(124, 247, 255, 0.3);
    padding-bottom: 0.3rem;
    margin: 1.2rem 0 0.6rem;
}

[data-testid="stForm"] {
    background: rgba(8, 12, 30, 0.75);
    border: 1px solid rgba(124, 247, 255, 0.35);
    border-radius: 10px;
    box-shadow: 0 0 25px rgba(124, 247, 255, 0.12);
}

.stButton button, [data-testid="stFormSubmitButton"] button {
    font-family: 'VT323', monospace !important;
    letter-spacing: 0.2em;
    font-size: 1.4rem !important;
    background: linear-gradient(90deg, #ffb347, #ffd75e) !important;
    color: #120a00 !important;
    border: none !important;
    box-shadow: 0 0 20px rgba(255, 215, 94, 0.5);
    width: 100%;
}

.status-pill {
    display: inline-block;
    font-family: 'VT323', monospace;
    font-size: 1rem;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    border: 1px solid currentColor;
}

.verdict {
    text-align: center;
    padding: 2rem 1.5rem;
    border-radius: 12px;
    margin-top: 1.5rem;
    animation: reveal 1.2s ease-out;
}
.verdict h2 { font-size: 2.4rem; margin: 0.3rem 0 0.8rem; }
.verdict p { line-height: 1.7; }
.verdict .big {
    font-family: 'VT323', monospace;
    font-size: 4rem;
    margin: 0.5rem 0;
}
.verdict.void {
    background: radial-gradient(circle at center, rgba(180, 124, 255, 0.35), rgba(20, 5, 40, 0.9));
    border: 1px solid var(--void);
    box-shadow: 0 0 45px rgba(180, 124, 255, 0.55);
}
.verdict.void h2, .verdict.void .big { color: var(--void); text-shadow: 0 0 14px var(--void); }
.verdict.stay {
    background: radial-gradient(circle at center, rgba(124, 247, 255, 0.25), rgba(5, 20, 35, 0.9));
    border: 1px solid var(--glow);
    box-shadow: 0 0 45px rgba(124, 247, 255, 0.4);
}
.verdict.stay h2, .verdict.stay .big { color: var(--glow); text-shadow: 0 0 14px var(--glow); }

/* hidden until its delay, then a gold cursor block for a moment, then plain text */
.typewriter span {
    animation: tw-show 0s linear both, tw-cursor 0.06s linear;
}
@keyframes tw-show {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes tw-cursor {
    from, to { background: var(--gold); color: var(--gold); }
}
.typewriter .tw-cursor {
    display: inline-block;
    width: 0.55em;
    height: 1em;
    vertical-align: text-bottom;
    background: var(--gold);
    opacity: 0;
    animation: blink 1s steps(1, end) infinite;
}
@keyframes blink {
    0%   { opacity: 1; }
    50%  { opacity: 0; }
}
@media (prefers-reduced-motion: reduce) {
    .typewriter span { animation: none; opacity: 1; }
    .typewriter .tw-cursor { animation: none; opacity: 1; }
}

@keyframes reveal {
    0%   { opacity: 0; transform: scale(0.85); filter: blur(8px); }
    60%  { opacity: 1; filter: blur(0); }
    100% { transform: scale(1); }
}

.footnote {
    text-align: center;
    font-size: 1rem;
    opacity: 0.6;
    margin-top: 1rem;
}
</style>
"""

st.markdown(SPACE_CSS, unsafe_allow_html=True)


PLANET_LORE = {
    "Earth": "🌍 Earth: crowded, tired, and hopeful for a fresh start.",
    "Europa": "🧊 Europa: the wealthy ice moon of old-money dynasties.",
    "Mars": "🔴 Mars: red dust, hard work, and stubborn pioneers.",
}

DESTINATION_LORE = {
    "TRAPPIST-1e": "TRAPPIST-1e: a temperate world under a dim red sun.",
    "55 Cancri e": "55 Cancri e: a scorching super-Earth with diamond skies.",
    "PSO J318.5-22": "PSO J318.5-22: a rogue planet drifting with no star at all.",
}

DECK_LORE = {
    "A": "A · Observation suites",
    "B": "B · First class",
    "C": "C · First class",
    "D": "D · Business class",
    "E": "E · Standard",
    "F": "F · Standard",
    "G": "G · Economy",
    "T": "T · Restricted area (??)",
}


try:
    response = requests.get(f"{api_base_url}/health", timeout=5)
    api_online = response.ok and response.json().get("status") == "ok"
except (requests.exceptions.RequestException, ValueError):
    api_online = False


st.markdown(
    """
    <div class="hero">
        <div class="sub">INTERSTELLAR LINER · YEAR 2912</div>
        <h1>SPACESHIP TITANIC</h1>
        <div class="sub">BOARDING TERMINAL 7</div>
    </div>
    """,
    unsafe_allow_html=True,
)

pill_color = "#7cf7ff" if api_online else "#ff6b6b"
pill_text = "● ORACLE ONLINE" if api_online else "● ORACLE OFFLINE"
st.markdown(
    f'<div style="text-align:center"><span class="status-pill" '
    f'style="color:{pill_color}">{pill_text}</span></div>',
    unsafe_allow_html=True,
)

INTRO = [
    ("13,000 passengers. One maiden voyage.", None),
    ("\n\n", None),
    ("Halfway to the new worlds, while rounding Alpha Centauri, the ", None),
    ("Spaceship Titanic", "b"),
    (" brushed against a hidden ", None),
    ("spacetime anomaly", 'b style="color:#b47cff"'),
    (" tucked inside a dust cloud. The ship survived intact. ", None),
    ("Almost half of its passengers did not.", "i"),
    (" They were transported into another dimension.", None),
    ("\n\n", None),
    ("You are standing at the boarding gate. Fill in your boarding pass, "
     "and the ship's oracle will tell you your fate.", None),
]


def typewriter(segments, char_delay=0.018):
    # one span per character, each revealed a little later than the last
    delay = 0.4
    parts = []
    for text, tag in segments:
        if text.startswith("\n"):
            parts.append("<br>" * len(text))
            delay += 0.5
            continue
        chars = []
        for char in text:
            chars.append(
                f'<span style="animation-delay:{delay:.2f}s">{escape(char)}</span>'
            )
            delay += char_delay
            if char in ".,":
                delay += 0.3 if char == "." else 0.1
        inner = "".join(chars)
        if tag:
            inner = f"<{tag}>{inner}</{tag.split()[0]}>"
        parts.append(inner)
    parts.append(
        f'<span class="tw-cursor" style="animation-delay:{delay:.2f}s"></span>'
    )
    return "".join(parts)


st.markdown(
    f'<div class="crawl typewriter">{typewriter(INTRO)}</div>',
    unsafe_allow_html=True,
)


if "passenger_id" not in st.session_state:
    group = random.randint(1, 9280)
    member = random.randint(1, 3)
    st.session_state.passenger_id = f"{group:04d}_{member:02d}"


with st.form("boarding_pass"):
    st.markdown('<div class="panel-title">▸ IDENTITY</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        traveller_name = st.text_input(
            "Your name (for the manifest)",
            value="",
            placeholder="Captain Nobody",
        )
    with col2:
        passenger_id = st.text_input(
            "Boarding pass ID",
            value=st.session_state.passenger_id,
            help="Format ####_##. The first four digits are your travel group.",
        )

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", min_value=0, max_value=80, value=28)
    with col2:
        vip = st.toggle("👑 VIP status")

    st.markdown('<div class="panel-title">▸ VOYAGE</div>', unsafe_allow_html=True)

    home_planet = st.radio(
        "Departing from",
        options=list(PLANET_LORE),
        format_func=lambda planet: PLANET_LORE[planet],
    )

    destination = st.radio(
        "Bound for",
        options=list(DESTINATION_LORE),
        format_func=lambda planet: DESTINATION_LORE[planet],
    )

    cryo_sleep = st.toggle(
        "❄️ Sleep through the voyage in a cryo-pod",
        help="Cryosleepers stay sealed in their cabins for the entire trip.",
    )

    st.markdown('<div class="panel-title">▸ QUARTERS</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        cabin_deck = st.selectbox(
            "Deck",
            options=list(DECK_LORE),
            index=5,
            format_func=lambda deck: DECK_LORE[deck],
        )
    with col2:
        cabin_number = st.number_input(
            "Cabin #", min_value=0, max_value=1894, value=412, step=1
        )
    with col3:
        cabin_side = st.selectbox(
            "Side",
            options=["P", "S"],
            format_func=lambda side: "Port" if side == "P" else "Starboard",
        )

    st.markdown(
        '<div class="panel-title">▸ ONBOARD SPENDING (GALACTIC CREDITS)</div>',
        unsafe_allow_html=True,
    )
    if cryo_sleep:
        st.caption("❄️ You'll be frozen, so you won't be spending anything.")

    col1, col2 = st.columns(2)
    with col1:
        room_service = st.slider("🛎️ Room service", 0, 5000, 0, step=50)
        food_court = st.slider("🍜 Food court", 0, 5000, 0, step=50)
        shopping_mall = st.slider("🛍️ Promenade shops", 0, 5000, 0, step=50)
    with col2:
        spa = st.slider("💆 Zero-G spa", 0, 5000, 0, step=50)
        vr_deck = st.slider("🕶️ VR deck", 0, 5000, 0, step=50)

    launch = st.form_submit_button("🚀  BOARD THE SHIP")


if launch:
    spending = [room_service, food_court, shopping_mall, spa, vr_deck]
    if cryo_sleep:
        spending = [0, 0, 0, 0, 0]

    payload = {
        "PassengerId": passenger_id,
        "HomePlanet": home_planet,
        "CryoSleep": cryo_sleep,
        "Cabin": f"{cabin_deck}/{cabin_number}/{cabin_side}",
        "Destination": destination,
        "Age": age,
        "VIP": vip,
        "RoomService": spending[0],
        "FoodCourt": spending[1],
        "ShoppingMall": spending[2],
        "Spa": spending[3],
        "VRDeck": spending[4],
    }

    name = traveller_name.strip() or "Passenger " + passenger_id

    with st.status("Departing...", expanded=True) as voyage:
        st.write(f"🛫 Leaving the orbit of {home_planet}...")
        time.sleep(0.8)
        st.write("🌌 Cruising at 0.6c. All systems nominal...")
        time.sleep(0.8)
        st.write("⚠️ Rounding Alpha Centauri. Sensors report an unidentified dust cloud...")
        time.sleep(1.0)
        st.write("🌀 **SPACETIME ANOMALY DETECTED**. Consulting the oracle...")

        try:
            prediction_response = requests.post(
                f"{api_base_url}/predict", json=payload, timeout=10
            )
            if prediction_response.status_code == 422:
                voyage.update(label="Boarding denied", state="error")
                st.error(
                    "The gate scanner rejected your boarding pass. "
                    "Check the ID format (####_##)."
                )
                st.stop()
            prediction_response.raise_for_status()
            prediction_data = prediction_response.json()
        except requests.exceptions.RequestException:
            voyage.update(label="Transmission lost", state="error")
            st.error(
                "The oracle is unreachable. Is the prediction API running at "
                f"{api_base_url}?"
            )
            st.stop()

        time.sleep(0.6)
        voyage.update(label="Voyage complete", state="complete", expanded=False)

    transported = prediction_data["transported"]
    probability = prediction_data["probability"]

    if transported:
        st.markdown(
            f"""
            <div class="verdict void">
                <div class="mono">MANIFEST UPDATE · {passenger_id}</div>
                <h2>YOU HAVE VANISHED</h2>
                <p>
                The anomaly folded around your cabin, and <b>{name}</b> was
                pulled into another dimension. Your seat on the shuttle to
                {destination} stays empty.
                </p>
                <div class="big">{probability:.0%}</div>
                <div class="mono">ANOMALY RESONANCE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verdict stay">
                <div class="mono">MANIFEST UPDATE · {passenger_id}</div>
                <h2>YOU REMAIN IN THIS DIMENSION</h2>
                <p>
                The anomaly passed through the hull, but <b>{name}</b> stayed
                put. You step off the ship at <b>{destination}</b>, shaken but
                whole, still wondering where everyone else went.
                </p>
                <div class="big">{probability:.0%}</div>
                <div class="mono">ANOMALY RESONANCE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="footnote">Anomaly resonance is the model\'s estimated '
        "probability of being transported. It is an estimate, not a "
        "prophecy.</div>",
        unsafe_allow_html=True,
    )
