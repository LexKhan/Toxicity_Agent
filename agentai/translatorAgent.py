from rag_setup import ToxicityRAG

class TranslatorAgent:
    def __init__(self, rag: ToxicityRAG):
        self.rag = rag
        print("   Translator ready (Qwen3)")

    def _build_prompt(self, content: str) -> str:
        return f"""You are a multilingual language detection and translation engine.

TASK:
1. Detect the language of the input text
2. If the text is NOT in English, translate it to English naturally
3. If the text IS in English, return it as-is
4. Preserve tone, emotion, and intent — do NOT sanitize or soften the meaning

IMPORTANT:
- Keep slang, insults, and informal language as close to the original intent as possible
- If the text mixes languages (code-switching), translate the non-English parts
- Do NOT explain, do NOT add commentary

Reply in EXACTLY this format:
DETECTED_LANGUAGE: <language name>
IS_ENGLISH: <YES or NO>
TRANSLATED: <translated text or original if already English>

Input text:
\"\"\"{content}\"\"\""""

    def translate(self, content: str) -> dict:
        prompt = self._build_prompt(content)
        raw_response = self.rag.llm_sarcasm.invoke(prompt)
        raw = raw_response.content if hasattr(raw_response, "content") else raw_response
        raw = raw.strip()

        # strip <think> block if present (Qwen3 reasoning model)
        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()

        result = {
            "detected_language": "unknown",
            "is_english":        True,
            "translated":        content,   # fallback to original
        }

        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("DETECTED_LANGUAGE:"):
                result["detected_language"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("IS_ENGLISH:"):
                result["is_english"] = line.split(":", 1)[1].strip().upper() == "YES"
            elif line.upper().startswith("TRANSLATED:"):
                result["translated"] = line.split(":", 1)[1].strip()

        translation_preview = "(English — no translation needed)" if result["is_english"] else f"→ {result['translated'][:80]}"
        print(f"     Translator: [{result['detected_language']}] {translation_preview}")
        return result