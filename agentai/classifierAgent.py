from rag_setup import ToxicityRAG
import re

class ClassifierAgent:
    def __init__(self, rag: ToxicityRAG):
        self.rag = rag
        print("   Classifier ready")

    def _build_prompt(self, content: str, sarcasm_result: dict) -> str:
        is_sarcasm = sarcasm_result["is_sarcasm"]
        meaning    = sarcasm_result["meaning"]

        sarcasm_note = ""
        if is_sarcasm == "sarcastic":
            sarcasm_note = (
                f"\nNOTE: The original text was sarcastic.\n"
                f"  Original:     \"{content}\"\n"
                f"  True meaning: \"{meaning}\"\n"
                f"Classify based on the TRUE MEANING, not the literal words.\n"
            )
        elif is_sarcasm == "ambiguous":
            sarcasm_note = (
                "\nNOTE: This text may or may not be sarcastic — context is unavailable.\n"
                "Classify at face value, but be aware the true intent is uncertain.\n"
            )

        text_to_classify = meaning if is_sarcasm == "sarcastic" else content  # Fix 2: was always `meaning`

        return f"""You are a strict content classification engine.

DEFINITIONS:
- TOXIC   : hate speech, threats, harassment, discrimination, personal attacks, obscene language
- NEUTRAL : factual statements, disagreements without hostility, questions, constructive criticism
- GOOD    : supportive, encouraging, appreciative, respectful, constructive communication
{sarcasm_note}
TEXT TO CLASSIFY:
\"\"\"{text_to_classify}\"\"\"

Reply with EXACTLY TWO LABELS separated by a hyphen. First, provide the main category (TOXIC, NEUTRAL, or GOOD). Second, provide the exact specific sub-label from the definitions above that explains why. 

Example formats: 
TOXIC - HATE SPEECH
NEUTRAL - DISAGREEMENTS WITHOUT HOSTILITY
GOOD - SUPPORTIVE

Reply with these LABELS ONLY. No extra punctuation other than the hyphen. No additional explanation."""

    def classify(self, content: str, sarcasm_result: dict) -> str:
        prompt = self._build_prompt(content, sarcasm_result)
        raw = self.rag.llm_qwen.invoke(prompt).strip().upper()
        match = re.search(r'\b(TOXIC|NEUTRAL|GOOD)\b\s*-\s*(.*)', raw)

        if match:
            TOXICITY  = match.group(1).strip()
            SUB_LABEL = match.group(2).strip()
        else:
            fallback_tox = re.search(r'\b(TOXIC|NEUTRAL|GOOD)\b', raw)
            TOXICITY = fallback_tox.group(1) if fallback_tox else "NEUTRAL"

            raw_cleaned = raw.replace(TOXICITY, '').strip(' -') if fallback_tox else ""
            cleaned_sub = re.sub(r'[^A-Z\s]', '', raw_cleaned).strip()
            SUB_LABEL = cleaned_sub if cleaned_sub else "UNKNOWN"

        print(f"     \nClassifier: {TOXICITY.capitalize()} - {SUB_LABEL.lower()}")
        return TOXICITY, SUB_LABEL