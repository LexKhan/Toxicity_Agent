from rag_setup import ToxicityRAG
import re

# Responsibility: Given raw text, retrieve the k most similar
# labeled examples from the FAISS vector store. Returns a list
# of dicts with 'content', 'classification', 'explanation'.
class RetrieverAgent:
    def __init__(self, rag: ToxicityRAG, k: int = 4):
        self.rag = rag
        self.k = k
        print("   RetrieverAgent ready")

    def retrieve(self, content: str) -> list:
        """
        Query FAISS for the k most semantically similar examples.
        Returns a list of example dicts.
        """
        docs = self.rag.vectorstore.similarity_search(content, k=self.k)

        examples = []
        for doc in docs:
            examples.append({
                "classification": doc.metadata.get("classification", "UNKNOWN"),
                "content":        doc.metadata.get("content", ""),
                "page_content":   doc.page_content,
            })

        print(f"    RetrieverAgent: found {len(examples)} similar examples "
              f"({[e['classification'] for e in examples]})")
        return examples