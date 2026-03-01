import streamlit as st
from pathlib import Path

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="ToxicityAgent",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load & inject CSS ─────────────────────────────────────────────────────────
def load_css(path: str) -> None:
    css = Path(path).read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css("styles.css")

# ── Try importing real agent; fall back to mock ───────────────────────────────
try:
    from agentai.agent import ToxicityAgent
    @st.cache_resource
    def get_agent():
        return ToxicityAgent()
    agent = get_agent()
    MOCK = False
except Exception:
    MOCK = True

def mock_analyze(text: str) -> dict:
    """Placeholder result used when the real agent is unavailable."""
    lower = text.lower()
    if any(w in lower for w in ["hate", "kill", "idiot", "stupid"]):
        cls, sub = "TOXIC", "Hate Speech"
    elif any(w in lower for w in ["okay", "fine", "whatever"]):
        cls, sub = "NEUTRAL", "Indifferent"
    else:
        cls, sub = "GOOD", "Positive"

    sarcasm = "no"
    meaning = ""
    if "totally" in lower or "sure" in lower:
        sarcasm = "ambiguous"
        meaning = "Possible ironic tone detected."

    return {
        "classification":    cls,
        "explanation":       "This is a mock response — connect the real ToxicityAgent to enable AI analysis.",
        "is_sarcasm":        sarcasm,
        "meaning":           meaning,
        "original":          text,
        "detected_language": "en",
        "translated":        None,
    }

def analyze(text: str) -> dict:
    return mock_analyze(text) if MOCK else agent.detect_and_respond(text)

# ── HTML builders ─────────────────────────────────────────────────────────────

def build_classifier_section(result: dict) -> str:
    cls      = result["classification"]          # TOXIC | NEUTRAL | GOOD
    sub      = result.get("sub_label", "—")      # sub-label from agent
    css_cls  = cls.lower()                        # maps to CSS class

    return f"""
    <div class="agent-section">
        <span class="agent-tag tag-classifier">Classifier</span>
        <br>
        <div class="classifier-bubble {css_cls}">
            <span class="tox-level">{cls}</span>
            <span class="tox-sublabel">{sub}</span>
        </div>
    </div>
    """

def build_sarcasm_section(result: dict) -> str:
    sarcasm  = result["is_sarcasm"]              # sarcastic | ambiguous | no
    meaning  = result.get("meaning", "")
    css_cls  = sarcasm.lower()

    label_map = {
        "sarcastic":  "SARCASM DETECTED",
        "ambiguous":  "AMBIGUOUS TONE",
        "no":         "NO SARCASM",
    }
    label = label_map.get(css_cls, "UNKNOWN")

    meaning_html = ""
    if meaning:
        meaning_html = f'<div class="sarcasm-meaning">↳ {meaning}</div>'

    return f"""
    <div class="agent-section">
        <span class="agent-tag tag-sarcasm">Sarcasm Detector</span>
        <br>
        <div class="sarcasm-bubble {css_cls}">
            <div>
                <span class="sarcasm-status">{label}</span>
                {meaning_html}
            </div>
        </div>
    </div>
    """

def build_responder_section(result: dict) -> str:
    explanation = result.get("explanation", "No response generated.")

    return f"""
    <div class="agent-section">
        <span class="agent-tag tag-responder">Responder</span>
        <br>
        <p class="responder-text">{explanation}</p>
    </div>
    """

def build_mother_container(result: dict) -> str:
    original  = result.get("original", "")
    lang      = result.get("detected_language", "en")
    translated = result.get("translated")

    classifier_html = build_classifier_section(result)
    sarcasm_html    = build_sarcasm_section(result)
    responder_html  = build_responder_section(result)

    # Truncate displayed input for header
    preview = (original[:72] + "…") if len(original) > 72 else original

    return f"""
    <div class="mother-container">
        <div class="mother-header">
            <span class="mother-header-label">INPUT</span>
            <span class="mother-header-input">{preview}</span>
        </div>
        {classifier_html}
        {sarcasm_html}
        {responder_html}
    </div>
    """

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of result dicts, newest first

# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">TOXICITY AGENT</div>
    <div class="page-subtitle">multi-agent content analysis system</div>
</div>
""", unsafe_allow_html=True)

if MOCK:
    st.markdown("""
    <div style="font-family:var(--font-pixel);font-size:0.38rem;color:#ffaa00;
                border:2px solid #ffaa00;padding:0.5rem 0.8rem;margin-bottom:1.2rem;">
        DEMO MODE &nbsp;|&nbsp; Real agent not found — showing mock responses
    </div>
    """, unsafe_allow_html=True)

# Input form
with st.form(key="analyze_form", clear_on_submit=False):
    user_input = st.text_area(
        "CONTENT TO ANALYZE",
        placeholder="Type or paste content here…",
        height=110,
    )
    submitted = st.form_submit_button("ANALYZE")

if submitted and user_input.strip():
    with st.spinner("Running pipeline…"):
        result = analyze(user_input.strip())
    st.session_state.history.insert(0, result)

# Divider
st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

# Render history
if st.session_state.history:
    for result in st.session_state.history:
        st.markdown(build_mother_container(result), unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state">
        <span class="empty-state-icon">[]</span>
        <p class="empty-state-text">
            NO ANALYSIS YET<br>
            ENTER TEXT ABOVE AND HIT ANALYZE
        </p>
    </div>
    """, unsafe_allow_html=True)