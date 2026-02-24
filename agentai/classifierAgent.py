from rag_setup import ToxicityRAG
import re

# Responsibility: Given the original text + retrieved examples,
# decide TOXIC / NEUTRAL / GOOD via a focused LLM call with a
# classification-only prompt. No explanation generated here.
class ClassifierAgent:
    def __init__(self, rag: ToxicityRAG):
        self.llm = rag.llm
        print("   ✅ ClassifierAgent ready")

    def _build_prompt(self, content: str, examples: list) -> str:
        example_block = ""
        for i, ex in enumerate(examples, 1):
            example_block += (
                f"  Example {i}:\n"
                f"    Text: {ex['content'][:120]}\n"
                f"    Label: {ex['classification']}\n\n"
            )

        return f"""You are a strict content classification engine.

DEFINITIONS:
- TOXIC   : hate speech, threats, harassment, discrimination, personal attacks, obscene language
- NEUTRAL : factual statements, disagreements without hostility, questions, constructive criticism
- GOOD    : supportive, encouraging, appreciative, respectful, constructive communication

SIMILAR LABELED EXAMPLES FOR REFERENCE:
{example_block}
TEXT TO CLASSIFY:
\"\"\"{content}\"\"\"

Reply with ONE word only — TOXIC, NEUTRAL, or GOOD. No punctuation. No explanation."""

    def classify(self, content: str, examples: list) -> str:
        prompt = self._build_prompt(content, examples)
        raw = self.llm.invoke(prompt).strip().upper()

        match = re.search(r'\b(TOXIC|NEUTRAL|GOOD)\b', raw)
        label = match.group(1) if match else "NEUTRAL"

        print(f"   ClassifierAgent: '{label}' (raw='{raw[:40]}')")
        return label