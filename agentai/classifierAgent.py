from rag_setup import ToxicityRAG
import re

# Responsibility: Given the original text + retrieved examples,
# decide TOXIC / NEUTRAL / GOOD via a focused LLM call with a
# classification-only prompt. No explanation generated here.
class ClassifierAgent:
    def __init__(self, rag: ToxicityRAG):
        self.llm = rag.llm_qwen
        print("   Classifier ready")

    def _build_prompt(self, content: str, sarcasm_result: dict, examples: list) -> str:
        example_block = ""
        for i, ex in enumerate(examples, 1):
            example_block += (
                f"  Example {i}:\n"
                f"    Text: {ex['content'][:120]}\n"
                f"    Label: {ex['classification']}\n\n"
            )

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

        return f"""You are a strict content classification engine.

DEFINITIONS:
- TOXIC   : hate speech, threats, harassment, discrimination, personal attacks, obscene language
- NEUTRAL : factual statements, disagreements without hostility, questions, constructive criticism
- GOOD    : supportive, encouraging, appreciative, respectful, constructive communication
{sarcasm_note}
SIMILAR LABELED EXAMPLES FOR REFERENCE:
{example_block}
TEXT TO CLASSIFY:
\"\"\"{meaning}\"\"\"

Reply with ONE word only — TOXIC, NEUTRAL, or GOOD. No punctuation. No explanation."""

    def classify(self, content: str, sarcasm_result: dict, examples: list) -> str:
        prompt = self._build_prompt(content, sarcasm_result, examples)
        raw = self.llm.invoke(prompt).strip().upper()
        match = re.search(r'\b(TOXIC|NEUTRAL|GOOD)\b', raw)
        label = match.group(1) if match else "NEUTRAL"
        return label