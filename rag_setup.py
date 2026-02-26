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
LLM_TEMPERATURE   = 0.2

class ToxicityRAG:
    def __init__(self, data_path: str = DEFAULT_DATA_PATH):
        self.data_path   = data_path
        self.vectorstore = None
        self.llm_qwen    = None
        self.llm_llama   = None

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

        required = ["classification", "content", "explanation"]
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
                f"Explanation: {row['explanation']}"
            )
            doc = Document(
                page_content=page_content,
                metadata={
                    "classification": row["classification"],
                    "content":        str(row["content"])[:200],
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

    def _connect_llm(self, model_name: str, temperature: float) -> OllamaLLM:
        print(f"   Connecting to Ollama ({model_name}) …")
        llm = OllamaLLM(model=model_name, temperature=temperature)
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

    def _build_llms(self):
        self.llm_qwen  = self._connect_llm(LLM_QWEN,  temperature=0.2)
        self.llm_llama = self._connect_llm(LLM_LLAMA, temperature=0.3)

    def _setup(self):
        embeddings       = self._build_embeddings()
        self.vectorstore = self._build_vectorstore(embeddings)
        self._build_llms()