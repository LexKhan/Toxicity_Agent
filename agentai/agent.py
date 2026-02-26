from rag_setup import ToxicityRAG
from .retrieverAgent  import RetrieverAgent
from .classifierAgent import ClassifierAgent
from .responderAgent  import ResponderAgent
from .sarcasmDetector import SarcasmDetector

class ToxicityAgent:
    def __init__(self):
        self.rag = ToxicityRAG()

        print("\n  Initialising agents …")
        self.retriever  = RetrieverAgent(self.rag, k=4)
        self.sarcasm    = SarcasmDetector(self.rag)
        self.classifier = ClassifierAgent(self.rag)
        self.responder  = ResponderAgent(self.rag)
        print("  All agents ready!\n")

    def detect_and_respond(self, content: str) -> dict:
        print(f"  PIPELINE START")
        print(f"  Input: {content[:100]}{'…' if len(content) > 100 else ''}\n")

        examples       = self.retriever.retrieve(content)
        sarcasm_result = self.sarcasm.detect(content, examples)
        classification = self.classifier.classify(content, sarcasm_result, examples)
        explanation    = self.responder.respond(content, classification, sarcasm_result)

        print(f"\n  Pipeline complete → {classification} (sarcasm: {sarcasm_result['is_sarcasm']})")

        return {
            "classification":     classification,
            "explanation":        explanation,
            "retrieved_examples": examples,
            "is_sarcasm":         sarcasm_result["is_sarcasm"],  # "no" | "ambiguous" | "sarcastic"
            "meaning":            sarcasm_result["meaning"],
        }

    def display_result(self, result: dict) -> None:
        colors = {"TOXIC": "\033[91m", "NEUTRAL": "\033[93m", "GOOD": "\033[92m"}
        icons  = {"TOXIC": "🔴", "NEUTRAL": "🟡", "GOOD": "🟢"}
        sarcasm_icons = {"no": "⚪", "ambiguous": "🟡", "sarcastic": "🟠"}
        reset  = "\033[0m"

        c = result["classification"]
        s = result["is_sarcasm"]

        print(f"\n  ANALYSIS RESULT")
        print(f"  {icons.get(c, '')} {colors.get(c, reset)}Classification: {c}{reset}")
        print(f"  {sarcasm_icons.get(s, '')} Sarcasm: {s}")

        if s == "sarcastic":
            print(f"  True meaning: {result['meaning']}")
        elif s == "ambiguous":
            print(f"  Note: sarcasm was ambiguous — classified at face value")

        print(f"\n  Responder: {result['explanation']}")


if __name__ == "__main__":
    agent = ToxicityAgent()