from rag_setup import ToxicityRAG
import re

# Responsibility: Given the text + confirmed classification,
# produce a human-readable explanation AND a message to the
# author (only for TOXIC content).
class ResponderAgent:
    def __init__(self, rag: ToxicityRAG):
        self.llm = rag.llm
        print("   ✅ ResponderAgent ready")

    def _build_prompt(self, content: str, classification: str) -> str:
        if classification == "TOXIC":
            extra = (
                "Also write a short, respectful 'Message to Author' (1–2 sentences) "
                "that tells the author why their comment is harmful and asks them to rephrase constructively."
            )
        else:
            extra = "Set 'Message to Author' to exactly: N/A"

        return f"""You are a content moderation assistant explaining a classification decision.

TEXT: \"\"\"{content}\"\"\"
CLASSIFICATION: {classification}

Write a clear 2–3 sentence Explanation of WHY this text is classified as {classification}.
Reference specific words or tone from the text.
{extra}

Use EXACTLY this format — no extra lines:
Explanation: <your explanation here>
Message to Author: <message or N/A>"""

    def respond(self, content: str, classification: str) -> dict:
        prompt = self._build_prompt(content, classification)
        raw = self.llm.invoke(prompt)

        explanation = ""
        message = "N/A"

        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("Explanation:"):
                explanation = line.replace("Explanation:", "").strip()
            elif line.startswith("Message to Author:"):
                message = line.replace("Message to Author:", "").strip()

        # Fallback if parsing failed
        if not explanation:
            explanation = f"This content was classified as {classification}."

        print(f"    ResponderAgent: explanation generated ({len(explanation)} chars)")
        return {
            "explanation":       explanation,
            "message_to_author": message,
        }