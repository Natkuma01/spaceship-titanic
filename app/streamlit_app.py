import os
import random
import time
from html import escape

import altair as alt
import requests
import streamlit as st

from model_metrics import (
    FALLBACK_MODELS,
    format_percent,
    holdout_chart_data,
    metrics_table_html,
    metrics_table_rows,
    models_from_api,
)


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

/* Bigger, easier-to-read text inside the passenger form only (was 14px). */
[data-testid="stForm"] [data-testid="stWidgetLabel"] p,
[data-testid="stForm"] [data-testid="stCheckbox"] p {
    font-size: 1.1rem !important;
}
[data-testid="stForm"] input,
[data-testid="stForm"] [data-baseweb="select"] *,
[data-testid="stForm"] [data-testid="stRadio"] [role="radiogroup"] p,
[data-testid="stForm"] [data-testid="stSlider"] [data-testid="stSliderThumbValue"],
[data-testid="stForm"] [data-testid="stSlider"] [data-testid="stSliderThumbValue"] p,
[data-testid="stForm"] [data-testid="stSlider"] [data-testid^="stSliderTickBar"] p,
[data-testid="stForm"] [data-testid="stCaptionContainer"],
[data-testid="stForm"] [data-testid="stCaptionContainer"] p,
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button p {
    font-size: 1.05rem !important;
}

/* Model picker: the radio options are drawn as three cards side by side. */
.st-key-oracle_picker [data-testid="stWidgetLabel"] p {
    font-size: 1.15rem !important;
}
.st-key-oracle_picker [role="radiogroup"] {
    display: grid !important;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.8rem;
    align-items: stretch !important;
    width: 100%;
}
/* each option (label + caption) is one card */
.st-key-oracle_picker [role="radiogroup"] > div {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    align-self: stretch;
    box-sizing: border-box;
    min-width: 0;
    margin: 0 !important;
    padding: 1rem 0.6rem 1.1rem;
    cursor: pointer;
    background: rgba(8, 12, 30, 0.75);
    border: 1px solid rgba(124, 247, 255, 0.35);
    border-radius: 10px;
    box-shadow: 0 0 18px rgba(124, 247, 255, 0.08);
    opacity: 0.7;
    transition: opacity 0.2s, border-color 0.2s, box-shadow 0.2s, transform 0.2s;
}
.st-key-oracle_picker [data-testid="stRadioOption"] {
    width: 100%;
    margin: 0 !important;
    padding: 0 !important;
    justify-content: center;
    cursor: pointer;
}
/* stretch the clickable label over the whole card, caption included */
.st-key-oracle_picker [data-testid="stRadioOption"]::after {
    content: "";
    position: absolute;
    inset: 0;
    z-index: 1;
    border-radius: 10px;
}
.st-key-oracle_picker [data-testid="stRadioOption"] > div {
    width: 100%;
    justify-content: center;
}
/* hide the radio dot; the whole card is the control */
.st-key-oracle_picker [data-testid="stRadioOption"] > div > div:first-child {
    display: none !important;
}
.st-key-oracle_picker [role="radiogroup"] > div:hover {
    opacity: 0.95;
    border-color: rgba(124, 247, 255, 0.7);
    box-shadow: 0 0 22px rgba(124, 247, 255, 0.25);
}
/* keyboard focus ring; the picked card already stands out, so skip it there */
.st-key-oracle_picker [role="radiogroup"] > div:has(input:focus-visible):not(:has(input:checked)) {
    outline: 2px dashed rgba(124, 247, 255, 0.8);
    outline-offset: 3px;
}
.st-key-oracle_picker [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p {
    color: var(--glow);
    font-size: 1.35rem !important;
    letter-spacing: 0.04em;
    overflow-wrap: anywhere;
    line-height: 1.2;
    text-align: center;
    margin: 0;
}
.st-key-oracle_picker [data-testid="stRadioCaption"] {
    width: 100%;
    padding: 0 !important;
    margin: 0.5rem 0 0 !important;
}
.st-key-oracle_picker [data-testid="stRadioCaption"] p {
    color: var(--ink);
    font-size: 1.05rem !important;
    line-height: 1.35;
    text-align: center;
    margin: 0;
}
.st-key-oracle_picker [role="radiogroup"] > div:has(input:checked) {
    opacity: 1;
    border: 1px solid var(--gold);
    background: rgba(40, 30, 5, 0.55);
    box-shadow: 0 0 0 1px var(--gold), 0 0 28px rgba(255, 215, 94, 0.5),
                inset 0 0 18px rgba(255, 215, 94, 0.12);
    transform: translateY(-2px);
}
.st-key-oracle_picker [role="radiogroup"] > div:has(input:checked) [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p {
    color: var(--gold);
    text-shadow: 0 0 12px rgba(255, 215, 94, 0.6);
}
.st-key-oracle_picker [data-testid="stCaptionContainer"],
.st-key-oracle_picker [data-testid="stCaptionContainer"] p {
    font-size: 1.05rem !important;
}
@media (max-width: 640px) {
    .st-key-oracle_picker [role="radiogroup"] {
        grid-template-columns: 1fr;
    }
}

.track-note {
    font-family: 'VT323', monospace;
    font-size: 1.1rem;
    color: var(--ink);
    opacity: 0.85;
    letter-spacing: 0.04em;
    margin: 0.2rem 0 1rem;
}
.track-legend {
    margin: 0.4rem 0 1rem;
}
.track-legend .track-note {
    margin: 0 0 0.15rem;
}
.track-legend .track-label {
    color: var(--glow);
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

.oracle-stat {
    font-family: 'VT323', monospace;
    color: var(--gold);
    font-size: 1.2rem;
    letter-spacing: 0.08em;
    margin: 0.2rem 0 1rem;
}

.metrics-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'VT323', monospace;
    font-size: 1.15rem;
    color: var(--ink);
    background: rgba(8, 12, 30, 0.75);
    border: 1px solid rgba(124, 247, 255, 0.35);
    margin-bottom: 0.8rem;
}
.metrics-table th, .metrics-table td {
    font-family: 'VT323', monospace !important;
    padding: 0.35rem 0.6rem;
    text-align: right;
    border-bottom: 1px solid rgba(124, 247, 255, 0.15);
}
.metrics-table th:first-child, .metrics-table td:first-child { text-align: left; }
.metrics-table th {
    color: var(--glow);
    letter-spacing: 0.12em;
    font-weight: normal;
}
.metrics-table tr.picked td {
    color: var(--gold);
    background: rgba(255, 215, 94, 0.1);
    text-shadow: 0 0 10px rgba(255, 215, 94, 0.45);
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


def fetch_models() -> dict:
    try:
        models_response = requests.get(f"{api_base_url}/models", timeout=5)
        models_response.raise_for_status()
        return models_from_api(models_response.json())
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError):
        return FALLBACK_MODELS


MODELS = fetch_models() if api_online else FALLBACK_MODELS


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


st.markdown('<div class="panel-title">▸ ORACLE</div>', unsafe_allow_html=True)

with st.container(key="oracle_picker"):
    model_key = st.radio(
        "🚨 Which oracle should read your fate? (*Required)",
        options=list(MODELS),
        index=None,
        horizontal=True,
        format_func=lambda key: MODELS[key]["display_name"],
        captions=[MODELS[key]["description"] for key in MODELS],
    )

    if model_key is None:
        st.caption("🔒 Choose a model above before you can board the ship.")
    else:
        picked_holdout = MODELS[model_key].get("metrics", {}).get("holdout_accuracy")
        if picked_holdout is not None:
            st.markdown(
                f'<div class="oracle-stat">◆ {escape(MODELS[model_key]["display_name"])} '
                f"holdout accuracy: {format_percent(picked_holdout)}</div>",
                unsafe_allow_html=True,
            )


def holdout_chart(data: list[dict]) -> alt.Chart:
    bars = (
        alt.Chart(alt.Data(values=data))
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "Holdout accuracy:Q",
                title="Holdout accuracy",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(tickCount=5, labelExpr="datum.value + '%'"),
            ),
            y=alt.Y(
                "Model:N",
                title=None,
                sort=None,
                axis=alt.Axis(labelLimit=220, ticks=False),
            ),
            color=alt.condition(
                alt.datum.Picked,
                alt.value("#ffd75e"),
                alt.value("#7cf7ff"),
            ),
            opacity=alt.condition(alt.datum.Picked, alt.value(1.0), alt.value(0.55)),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip(
                    "Holdout accuracy:Q", format=".1f", title="Holdout accuracy (%)"
                ),
            ],
        )
    )
    labels = bars.mark_text(align="left", dx=5, font="VT323", fontSize=17).encode(
        text=alt.Text("Holdout accuracy:Q", format=".1f"),
        color=alt.condition(alt.datum.Picked, alt.value("#ffd75e"), alt.value("#d9e6ff")),
        opacity=alt.value(1.0),
        tooltip=alt.value(None),
    )
    return (
        (bars + labels)
        .properties(height=alt.Step(34))
        .configure(background="transparent", font="VT323")
        .configure_axis(
            labelColor="#d9e6ff",
            titleColor="#7cf7ff",
            labelFontSize=16,
            titleFontSize=15,
            gridColor="rgba(124, 247, 255, 0.12)",
            domainColor="rgba(124, 247, 255, 0.35)",
        )
        .configure_view(strokeWidth=0)
    )


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
        age = st.slider("Age", min_value=0, max_value=120, value=28)
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

    cryo_sleep = st.toggle("❄️ Sleep through the voyage in a cryo-pod")

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

    launch = st.form_submit_button(
        "🚀  BOARD THE SHIP",
        disabled=model_key is None,
    )


TRACK_LEGEND = [
    ("CV accuracy", "average score from 5-fold cross validation on the training data."),
    ("Holdout accuracy", "share of correct guesses on 20% of passengers held back from training."),
    ("Holdout ROC-AUC", "how well the model ranks transported above not transported (1.0 is perfect)."),
]


def track_legend_html():
    lines = "".join(
        f'<div class="track-note"><span class="track-label">{label}:</span> {text}</div>'
        for label, text in TRACK_LEGEND
    )
    return f'<div class="track-legend">{lines}</div>'


# The verdict goes right under the form; the track record sits below it.
results = st.container()

table_rows = metrics_table_rows(MODELS)
if any(row["Holdout accuracy"] != "—" for row in table_rows):
    st.markdown(
        '<div class="panel-title">▸ ORACLE TRACK RECORD</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="track-note">How often each oracle guessed right on '
        "passengers it never saw.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(metrics_table_html(table_rows, model_key), unsafe_allow_html=True)
    chart_data = holdout_chart_data(MODELS, model_key)
    if chart_data:
        st.altair_chart(holdout_chart(chart_data), width="stretch")
    st.markdown(track_legend_html(), unsafe_allow_html=True)


with results:
    if launch and model_key is None:
        st.warning("Choose a model above before boarding.")
        st.stop()

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
            "model": model_key,
        }

        name = escape(traveller_name.strip() or "Passenger " + passenger_id)

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
            except requests.exceptions.Timeout:
                voyage.update(label="Transmission lost", state="error")
                st.error("The prediction API took too long to respond.")
                st.stop()
            except requests.exceptions.ConnectionError:
                voyage.update(label="Transmission lost", state="error")
                st.error("Cannot connect to the prediction API.")
                st.stop()
            except (requests.exceptions.RequestException, ValueError):
                voyage.update(label="Transmission lost", state="error")
                st.error("The prediction could not be completed.")
                st.stop()

            time.sleep(0.6)
            voyage.update(label="Voyage complete", state="complete", expanded=False)

        transported = prediction_data["transported"]
        probability = prediction_data["probability"]
        used_model = prediction_data.get("model_name", model_key)
        oracle_name = escape(
            MODELS.get(used_model, {}).get("display_name", used_model)
        )

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
                    <div class="mono">ORACLE · {oracle_name}</div>
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
                    <div class="mono">ORACLE · {oracle_name}</div>
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
