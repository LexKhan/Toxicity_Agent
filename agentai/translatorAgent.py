from rag_setup import ToxicityRAG
import re

# Responsibility: translates languages or mixed languages

class translatorAgent:
    def __init__(self, rag: ToxicityRAG):
        self.rag = rag 
        print("   Translator ready")

    def _build_prompt(self, content: str) -> str:
        return f"""You are a translation assistant. Your only job is to translate the given text into natural, fluent English.

RULES:
- If the text is already fully in English, return it as-is without any changes.
- If the text is in another language, translate it fully to English.
- If the text is mixed language (e.g. Tagalog + English, Malay + English), translate the non-English parts and produce a single clean English output.
- Preserve the original tone and intent — do NOT soften, censor, or alter the meaning.
- Do NOT add explanations, comments, or any extra text.
- Return ONLY the translated text, nothing else.

TEXT TO TRANSLATE:
\"\"\"{content}\"\"\"

TRANSLATED OUTPUT:"""
    
    def translate(self, content: str) -> str:
        prompt = self._build_prompt(content)
        raw = self.rag.llm_sailor.invoke(prompt)
        result = raw.content if hasattr(raw, "content") else str(raw)
        result = result.strip()
        self.rag.release_sailor()  
        
        if not result:
            print("   ⚠ Translator returned empty — using original text")
            return content

        print("\n" + "-"*60)
        print(f"   TranslatorAgent:")
        print(f"   Original:    {content[:80]}")
        print(f"   Translated:  {result[:80]}")
        print("-"*60)

        return result



