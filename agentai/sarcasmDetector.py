from rag_setup import ToxicityRAG
import re

class SarcasmDetector:
    def __init__(self, rag: ToxicityRAG):
        self.rag = rag   
        print("   SarcasmDetector ready")

    def _build_prompt(self, content: str, examples: list) -> str:
        example_block = ""
        for i, ex in enumerate(examples, 1):
            example_block += (
                f"  Example {i}:\n"
                f"    Text: {ex['content'][:120]}\n"
                f"    Label: {ex['classification']}\n\n"
            )

        return f"""You are an expert at detecting sarcasm and irony in text.

Sarcasm means the LITERAL words differ from (and are often the opposite of) the TRUE MEANING. 
While sarcasm can be mocking or harmful, it can also be completely NEUTRAL, humorous, or self-deprecating.

Because you are analyzing SINGLE messages without conversation history, some messages will be impossible to judge. If the text is ambiguous and could easily be genuine OR sarcastic depending on context, classify it as UNKNOWN.

Examples of varying sarcasm and ambiguity:
  - "Oh great, another flat tire." → YES (true meaning: "This is terrible.")
  - "Lovely weather for a picnic." (during a storm) → YES (true meaning: "The weather is awful.")
  - "I really loved how Bob handled that meeting." → UNKNOWN (could be literal praise or deadpan sarcasm)
  - "Thanks for nothing." → YES (true meaning: "I am upset you didn't help.")

TOXICITY SCALE:
  - GOOD    : positive, kind, constructive
  - NEUTRAL : no harmful or positive intent
  - TOXIC   : hateful, harmful, or offensive in true meaning

SIMILAR EXAMPLES FROM DATABASE FOR CONTEXT:
{example_block}

TEXT TO ANALYSE:
\"\"\"{content}\"\"\"

Respond in EXACTLY this format — no extra lines:
IS_SARCASTIC: [YES/NO/UNKNOWN]
TOXICITY: [GOOD/NEUTRAL/TOXIC] (based on TRUE meaning, not literal words)
TRUE_MEANING: [If YES: what the text ACTUALLY means. If NO or UNKNOWN: same as original text.]
        """

    def detect(self, content: str, examples: list) -> dict:
        prompt = self._build_prompt(content, examples)
        raw_response = self.llm_llama.invoke(prompt)
        raw = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
        self.rag.release_llama()

        # Parse response
        is_sarcasm = "no"       # "no" | "ambiguous" | "sarcastic"
        toxicity = "NEUTRAL"    # "GOOD" | "NEUTRAL" | "TOXIC" — always reflects true meaning
        meaning = content

        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("IS_SARCASTIC:"):
                val = line.replace("IS_SARCASTIC:", "").strip().upper()
                if val == "YES":
                    is_sarcasm = "sarcastic"
                elif val == "UNKNOWN":
                    is_sarcasm = "ambiguous"
            elif line.startswith("TOXICITY:"):
                toxicity = line.replace("TOXICITY:", "").strip().upper()
            elif line.startswith("TRUE_MEANING:"):
                meaning = line.replace("TRUE_MEANING:", "").strip() or content

        match is_sarcasm:
            case "sarcastic":
                print("\n" + "-"*60)
                print(f"     SarcasmDetector: SARCASTIC ({toxicity})")
                print(f"     Original: {content[:80]}")
                print(f"     Meaning:  {meaning[:80]}")
                print("-"*60)
            case "ambiguous":
                print("\n" + "-"*60)
                print(f"     SarcasmDetector: AMBIGUOUS ({toxicity})")
                print(f"     Original: {content[:80]}")
                print("-"*60)
            case _:
                print("\n" + "-"*60)
                print(f"     SarcasmDetector: no sarcasm ({toxicity})")
                print("-"*60)
        
        return {
            "is_sarcasm": is_sarcasm,  # "no" | "ambiguous" | "sarcastic"
            "toxicity":   toxicity,    # "GOOD" | "NEUTRAL" | "TOXIC" — based on true meaning
            "meaning":    meaning,
        }
    

