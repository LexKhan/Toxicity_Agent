from rag_setup import ToxicityRAG
from .retrieverAgent import RetrieverAgent
from .classifierAgent import ClassifierAgent
from .responderAgent import ResponderAgent

import re

# ORCHESTRATOR
# Wires the three agents together into a single pipeline and
# exposes the same public API as before so app.py and main.py
# require zero changes.
class ToxicityAgent:
    def __init__(self):
        self.rag = ToxicityRAG()

        print("\n  Initialising agents …")
        self.retriever  = RetrieverAgent(self.rag, k=4)
        self.classifier = ClassifierAgent(self.rag)
        self.responder  = ResponderAgent(self.rag)
        print("  All 3 agents ready!")

    def detect_and_respond(self, content: str) -> dict:
        print(f"  PIPELINE START")
        print(f"  Input: {content[:100]}{'…' if len(content) > 100 else ''}")

        examples = self.retriever.retrieve(content)
        classification = self.classifier.classify(content, examples)
        response = self.responder.respond(content, classification)

        result = {
            "classification":    classification,
            "explanation":       response["explanation"],
            "message_to_author": response["message_to_author"],
            "retrieved_examples": examples,   # bonus data available to UI
        }

        print(f"\n  Pipeline complete → {classification}")
        return result

    def display_result(self, result: dict) -> None:
        colors = {
            "TOXIC":   "\033[91m",
            "NEUTRAL": "\033[93m",
            "GOOD":    "\033[92m",
            "UNKNOWN": "\033[0m",
        }

        icons = {"TOXIC": "🔴", "NEUTRAL": "🟡", "GOOD": "🟢", "UNKNOWN": "⚪"}
        reset = "\033[0m"

        c = result["classification"]

        print(f"  ANALYSIS RESULT")
        print(f"  {icons.get(c, '')} {colors.get(c, reset)}Classification: {c}{reset}\n")

if __name__ == "__main__":
    agent = ToxicityAgent()

