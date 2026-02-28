from rag_setup import ToxicityRAG
import re

# Responsibility: Given raw text, retrieve the k most similar
# labeled examples from the FAISS vector store. Returns a list
# of dicts with 'content', 'classification', 'explanation'.
class RetrieverAgent:
    def __init__(self, rag: ToxicityRAG, k: int = 4):
        self.rag = rag
        self.k = k
        print("   Retriever ready")

    def retrieve(self, content: str) -> list:
        if self.rag.vectorstore is None:
            raise RuntimeError("Vectorstore is None — RAG setup may have failed.")

        docs = self.rag.vectorstore.similarity_search(content, k=self.k)

        examples = []
        for doc in docs:
            examples.append({
                "classification":    doc.metadata.get("classification",    "UNKNOWN"),
                "content":           doc.metadata.get("content",           ""),
                "explanation":       doc.metadata.get("explanation",       ""),
                "message_to_author": doc.metadata.get("message_to_author", "N/A"),  # ← added
                "page_content":      doc.page_content,
            })

        print("\n" + "-"*60)
        print(f"    RetrieverAgent: found {len(examples)} similar examples:")
        for i, e in enumerate(examples, 1):
            print(f"     [{i}] {e['classification']} — {e['content'][:80]}…")
        print("-"*60)

        return examples