from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain.docstore.document import Document

import pandas as pd
import os

FAISS_INDEX_PATH  = "faiss_index"
DEFAULT_DATA_PATH = "data/toxicity_examples.csv"
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

LLM_QWEN          = "qwen2.5:7b"
LLM_LLAMA         = "llama3.1:8b"
LLM_SAILOR        = "sailor2:8b"
LLM_TEMPERATURE   = 0.2

CTX_WINDOWS = {
    LLM_SAILOR: 4096,   # translation needs longer input/output
    LLM_LLAMA:  2048,   # sarcasm detection, shorter context fine
    LLM_QWEN:   2048,   # classification + verdict, 7B tight on 6GB
}

KEEP_ALIVE        = "0"           # Unload model from VRAM immediately after use
os.environ["OLLAMA_KEEP_ALIVE"] = KEEP_ALIVE

class ToxicityRAG:
    #
    # LLMs
    #
    @property
    def llm_sailor(self) -> OllamaLLM:
        if self._llm_sailor is None:
            self._llm_sailor = self._connect_llm(LLM_SAILOR, temperature=0.3)
        return self._llm_sailor

    @property
    def llm_llama(self) -> OllamaLLM:
        if self._llm_llama is None:
            self._llm_llama = self._connect_llm(LLM_LLAMA, temperature=0.3)
        return self._llm_llama

    @property
    def llm_qwen(self) -> OllamaLLM:
        if self._llm_qwen is None:
            self._llm_qwen = self._connect_llm(LLM_QWEN, temperature=LLM_TEMPERATURE)
        return self._llm_qwen

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

    def _connect_llm(self, model_name: str, temperature: float) -> OllamaLLM:
        ctx = CTX_WINDOWS.get(model_name, 2048)     # per-model context window
        print(f"   Connecting to Ollama ({model_name}, ctx={ctx}) …")
        llm = OllamaLLM(
            model=model_name,
            temperature=temperature,
            num_ctx=ctx,
            keep_alive=KEEP_ALIVE,
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

    def _setup(self):
        embeddings       = self._build_embeddings()
        self.vectorstore = self._build_vectorstore(embeddings)