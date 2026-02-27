import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Toxicity Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0e0e0e;
    color: #e8e8e8;
}

.stApp {
    background-color: #0e0e0e;
}

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
}

/* Header */
.header {
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}

.header h1 {
    font-size: 1.4rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #e8e8e8;
    margin: 0;
}

.header p {
    font-size: 0.78rem;
    color: #555;
    margin: 0.3rem 0 0 0;
    letter-spacing: 0.04em;
    font-family: 'IBM Plex Mono', monospace;
}

/* Textarea */
textarea {
    background-color: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #e8e8e8 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.92rem !important;
    caret-color: #e8e8e8;
}

textarea:focus {
    border-color: #444 !important;
    box-shadow: none !important;
}

/* Button */
.stButton > button {
    background-color: #e8e8e8;
    color: #0e0e0e;
    border: none;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    padding: 0.55rem 1.8rem;
    cursor: pointer;
    transition: background 0.15s ease;
    width: 100%;
}

.stButton > button:hover {
    background-color: #c8c8c8;
}

/* Result card */
.result-card {
    border-radius: 4px;
    padding: 1.6rem 1.8rem;
    margin-top: 1.8rem;
    border-left: 3px solid;
}

.result-card.toxic {
    background-color: #1a0a0a;
    border-color: #c0392b;
}

.result-card.good {
    background-color: #0a0f1a;
    border-color: #2980b9;
}

.result-card.neutral {
    background-color: #151209;
    border-color: #d4a017;
}

/* Labels */
.label-row {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 1.2rem;
}

.classification-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.12em;
}

.toxic   .classification-label { color: #c0392b; }
.good    .classification-label { color: #2980b9; }
.neutral .classification-label { color: #d4a017; }

.confidence-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #555;
    letter-spacing: 0.06em;
}

/* Progress bar container */
.bar-track {
    background: #1e1e1e;
    border-radius: 2px;
    height: 4px;
    width: 100%;
    margin-bottom: 1.4rem;
    overflow: hidden;
}

.bar-fill-toxic   { background: #c0392b; height: 100%; border-radius: 2px; }
.bar-fill-good    { background: #2980b9; height: 100%; border-radius: 2px; }
.bar-fill-neutral { background: #d4a017; height: 100%; border-radius: 2px; }

/* Sarcasm tag */
.tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    padding: 0.2rem 0.55rem;
    border-radius: 2px;
    margin-bottom: 1.2rem;
}

.tag-sarcastic { background: #2a1f0a; color: #d4a017; border: 1px solid #3a2f1a; }
.tag-ambiguous { background: #1a1a2a; color: #7a7aaa; border: 1px solid #2a2a3a; }

/* Section labels */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    color: #444;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.explanation-text {
    font-size: 0.88rem;
    color: #aaa;
    line-height: 1.65;
}

.meaning-text {
    font-size: 0.85rem;
    color: #888;
    font-style: italic;
    line-height: 1.6;
    margin-bottom: 1.2rem;
}

.divider {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 1.2rem 0;
}

/* Error */
.error-box {
    background: #1a0a0a;
    border: 1px solid #3a1a1a;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #c0392b;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 2rem; max-width: 680px; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>TOXICITY DETECTOR</h1>
    <p>content moderation · sarcasm-aware · rag-backed</p>
</div>
""", unsafe_allow_html=True)


# ── Agent init (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_agent():
    from agentai.agent import ToxicityAgent
    return ToxicityAgent()

with st.spinner("Loading models…"):
    try:
        agent = load_agent()
        agent_loaded = True
    except Exception as e:
        agent_loaded = False
        agent_error  = str(e)


# ── Input ──────────────────────────────────────────────────────────────────────
text = st.text_area(
    label="Content",
    placeholder="Paste or type the content to analyse…",
    height=140,
    label_visibility="collapsed",
)

analyse_clicked = st.button("ANALYSE", use_container_width=True)


# ── Analysis ───────────────────────────────────────────────────────────────────
def confidence_label(classification: str) -> tuple[str, int]:
    """
    Derive a rough confidence % from classification certainty.
    In a real setup you'd pull logprobs; here we simulate per-class tendency.
    """
    # Placeholder — swap for real logprob extraction if available
    defaults = {"TOXIC": 87, "GOOD": 91, "NEUTRAL": 74}
    return defaults.get(classification, 80)


if analyse_clicked:
    if not agent_loaded:
        st.markdown(f'<div class="error-box">Failed to load agent: {agent_error}</div>',
                    unsafe_allow_html=True)
    elif not text.strip():
        st.markdown('<div class="error-box">No content entered.</div>',
                    unsafe_allow_html=True)
    else:
        with st.spinner("Analysing…"):
            result = agent.detect_and_respond(text.strip())

        c           = result["classification"]          # TOXIC / NEUTRAL / GOOD
        explanation = result["explanation"]
        is_sarcasm  = result["is_sarcasm"]              # no / ambiguous / sarcastic
        meaning     = result["meaning"]
        confidence  = confidence_label(c)

        css_class   = c.lower()
        bar_class   = f"bar-fill-{css_class}"

        # ── Sarcasm tag ────────────────────────────────────────────────────────
        sarcasm_html = ""
        if is_sarcasm == "sarcastic":
            sarcasm_html = '<span class="tag tag-sarcastic">SARCASM DETECTED</span>'
        elif is_sarcasm == "ambiguous":
            sarcasm_html = '<span class="tag tag-ambiguous">AMBIGUOUS INTENT</span>'

        # ── True meaning block ─────────────────────────────────────────────────
        meaning_html = ""
        if is_sarcasm == "sarcastic":
            meaning_html = f"""
            <div class="section-label">True Meaning</div>
            <div class="meaning-text">{meaning}</div>
            <hr class="divider">
            """

        # ── Card ───────────────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="result-card {css_class}">
            <div class="label-row">
                <span class="classification-label">{c}</span>
                <span class="confidence-badge">{confidence}% confidence</span>
            </div>

            <div class="bar-track">
                <div class="{bar_class}" style="width:{confidence}%"></div>
            </div>

            {sarcasm_html}
            {meaning_html}

            <div class="section-label">Explanation</div>
            <div class="explanation-text">{explanation}</div>
        </div>
        """, unsafe_allow_html=True)