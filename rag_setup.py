from langchain_ollama import OllamaLLM
import os
from enum import Enum

class LLMProvider(str, Enum):
    LOCAL = "local"   # Ollama (demo)
    GROQ  = "groq"    # Groq API (production)

ACTIVE_PROVIDER = LLMProvider.LOCAL

MODELS = {
    LLMProvider.LOCAL: {
        "qwen":  "qwen2.5:7b",
        "llama": "llama3.1:8b",
    },
    LLMProvider.GROQ: {
        "qwen":  "qwen3-32b",               # TODO: confirm exact Groq model string
        "llama": "llama-3.3-70b-versatile",
    },
}

LLM_QWEN  = MODELS[ACTIVE_PROVIDER]["qwen"]
LLM_LLAMA = MODELS[ACTIVE_PROVIDER]["llama"]

AGENT_MODELS = {
    "sarcasm":    LLM_QWEN,
    "classifier": LLM_LLAMA,
    "responder":  LLM_LLAMA,
}

LLM_TEMPERATURE   = 0.2

CTX_WINDOWS = {
    LLM_QWEN:  2048,
    LLM_LLAMA: 2048,
}

class ToxicityRAG:
    #
    # LLMs
    #

    def __init__(self):
        self._llm_qwen  = None
        self._llm_llama = None

    @property
    def llm_qwen(self) -> OllamaLLM:
        if self._llm_qwen is None:
            self._llm_qwen = self._connect_llm(LLM_QWEN)
        return self._llm_qwen

    @property
    def llm_llama(self) -> OllamaLLM:
        if self._llm_llama is None:
            self._llm_llama = self._connect_llm(LLM_LLAMA)
        return self._llm_llama

    @property
    def llm_sarcasm(self) -> OllamaLLM:
        return self.llm_qwen

    @property
    def llm_classifier(self) -> OllamaLLM:
        return self.llm_llama

    @property
    def llm_responder(self) -> OllamaLLM:
        return self.llm_llama

    def _release(self, attr: str, label: str):
        """Generic — evicts model from VRAM and nulls the Python reference."""
        llm = getattr(self, attr, None)
        if llm is not None:
            try:
                llm.invoke("", options={"num_predict": 0})  # flush VRAM
            except Exception:
                pass
            setattr(self, attr, None)
            print(f"   ✓ {label} unloaded from VRAM")

    def release_sailor(self): self._release("_llm_sailor", "Sailor2")
    def release_llama(self):  self._release("_llm_llama",  "LLaMA")
    def release_qwen(self):   self._release("_llm_qwen",   "Qwen")

    def _connect_llm(self, model_name: str) -> OllamaLLM:
        """
        Currently wires up Ollama for local demo.
        TODO: when ACTIVE_PROVIDER == LLMProvider.GROQ, initialise
              ChatGroq here instead and return it.
        """
        if ACTIVE_PROVIDER == LLMProvider.GROQ:
            # Placeholder — swap in when setting up Groq:
            #
            #   from langchain_groq import ChatGroq
            #   return ChatGroq(
            #       model=model_name,
            #       temperature=LLM_TEMPERATURE,
            #       api_key=os.environ["GROQ_API_KEY"],
            #   )
            raise NotImplementedError(
                "Groq provider selected but not yet configured. "
                "Set ACTIVE_PROVIDER = LLMProvider.LOCAL for local demo."
            )

        ctx = CTX_WINDOWS.get(model_name, 2048)
        print(f"   Connecting to Ollama ({model_name}, ctx={ctx}) …")
        llm = OllamaLLM(
            model=model_name,
            temperature=LLM_TEMPERATURE,
            num_ctx=ctx,
        )
        try:
            llm.invoke("ping")
            print(f"   ✓ {model_name} connected")
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Ollama model '{model_name}': {e}\n"
                f"Make sure Ollama is running and the model is pulled:\n"
                f"  ollama pull {model_name}"
            ) from e
        return llm

