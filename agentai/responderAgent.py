from rag_setup import ToxicityRAG
import re

# Responsibility: Given the text + confirmed classification,
# produce a human-readable explanation AND a message to the
# author (only for TOXIC content).
class ResponderAgent:
    def __init__(self, rag: ToxicityRAG):
        self.rag = rag 
        print("   Responder ready")

    def _build_prompt(self, content: str, classification: str, sub_label:str, sarcasm_result: dict) -> str:
        is_sarcasm = sarcasm_result["is_sarcasm"]
        meaning    = sarcasm_result["meaning"]

        sarcasm_context = ""
        if is_sarcasm == "sarcastic":
            sarcasm_context = (
                f"\nThis text was identified as SARCASTIC.\n"
                f"True meaning: \"{meaning}\"\n"
                f"Mention the sarcastic nature in your explanation.\n"
            )
        elif is_sarcasm == "ambiguous":
            sarcasm_context = (
                "\nThis text may be sarcastic — intent is uncertain without more context.\n"
                "Mention this ambiguity in your explanation.\n"
            )

        return f"""You are a content moderation assistant explaining a classification decision.

ORIGINAL TEXT: \"\"\"{content}\"\"\"
TOXICITY_LEVEL: {classification}
TYPE: {sub_label}
{sarcasm_context}
Write a clear 2–3 sentence explanation of WHY this text is classified as {classification} and {sub_label}.
Reference specific words or tone from the text.

Respond in EXACTLY this format — no extra lines:
Explanation: [your explanation]"""

    def respond(self, content: str, classification: str, sub_label: str, sarcasm_result: dict) -> str:
        prompt = self._build_prompt(content, classification, sub_label, sarcasm_result)
        raw = self.rag.llm_qwen.invoke(prompt)
        self.rag.release_qwen()

        explanation = ""
        message = "N/A"

        explanation = ""
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("Explanation:"):
                explanation = line.replace("Explanation:", "").strip()
                break

        if not explanation:
            explanation = f"This content was classified as {classification}."

        return explanation