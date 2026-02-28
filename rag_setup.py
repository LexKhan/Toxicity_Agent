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
    def __init__(self, data_path: str = DEFAULT_DATA_PATH):
        self.data_path   = data_path
        self.vectorstore = None
        self._llm_sailor  = None    
        self._llm_llama   = None    
        self._llm_qwen    = None 

        print("Initialising shared RAG backend …")
        self._setup()
        print("RAG backend ready!\n")

    def _load_documents(self) -> list:
        print("    Loading examples from CSV …")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found at '{self.data_path}'.\n"
                "Run:  python preprocess_data.py"
            )

        df = pd.read_csv(self.data_path)

        required = ["classification", "content", "explanation", "message_to_author"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        df = df.dropna(subset=["content", "classification"]).reset_index(drop=True)
        print(f"   ✓ {len(df)} examples loaded")

        documents = []
        for _, row in df.iterrows():
            page_content = (
                f"Classification: {row['classification']}\n"
                f"Content: {row['content']}\n"
                f"Explanation: {row['explanation']}\n"
                f"Message: {row['message_to_author']}"   # ← added
            )
            doc = Document(
                page_content=page_content,
                metadata={
                    "classification":    row["classification"],
                    "content":           str(row["content"])[:200],
                    "explanation":       str(row["explanation"]),
                    "message_to_author": str(row["message_to_author"]),  # ← added
                },
            )
            documents.append(doc)

        return documents

    def _build_embeddings(self) -> HuggingFaceEmbeddings:
        print(f"   Loading embedding model ({EMBEDDING_MODEL}) …")
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _build_vectorstore(self, embeddings: HuggingFaceEmbeddings):
        if os.path.exists(FAISS_INDEX_PATH):
            print(f"   Loading FAISS index from cache ({FAISS_INDEX_PATH}) …")
            try:
                vs = FAISS.load_local(
                    FAISS_INDEX_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                print("   ✓ FAISS index loaded from cache")
                return vs
            except Exception as e:
                print(f"   ⚠ Cache load failed ({e}), rebuilding …")

        print("    Building FAISS index from documents …")
        documents = self._load_documents()
        vs = FAISS.from_documents(documents=documents, embedding=embeddings)

        try:
            vs.save_local(FAISS_INDEX_PATH)
            print(f"   ✓ FAISS index saved to '{FAISS_INDEX_PATH}'")
        except Exception as e:
            print(f"   ⚠ Could not save FAISS index: {e}")

        return vs

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