from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_community.chains import RetrievalQA

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain.docstore.document import Document

import pandas as pd
import os

class ToxicityRAG:
    def __init__(self, data_path="data/toxicity_examples.csv"):
        """Initialize RAG system for toxicity detection"""
        self.data_path = data_path
        self.vectorstore = None
        self.qa_chain = None
        
        print("🔧 Initializing Toxicity Detection RAG System...")
        self.setup_rag()
        print("✅ RAG System Ready!\n")
    
    def load_csv_as_documents(self):
        """Load CSV and convert to LangChain documents"""
        print("   📄 Loading toxicity examples from CSV...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}. Please run preprocess_data.py first.")
        
        df = pd.read_csv(self.data_path)
        print(f"   ✓ Loaded {len(df)} examples")
        
        documents = []
        for _, row in df.iterrows():
            content = f"""Classification: {row['classification']}
Content: {row['content']}
Explanation: {row['explanation']}
Message to Author: {row['message_to_author']}"""
            
            doc = Document(
                page_content=content,
                metadata={
                    'classification': row['classification'],
                    'content': row['content'][:100]
                }
            )
            documents.append(doc)
        
        print(f"   ✓ Created {len(documents)} document embeddings")
        return documents
    
    def setup_rag(self):
        """Initialize RAG system with embeddings and LLM"""
        documents = self.load_csv_as_documents()
        
        print("   🧠 Creating embeddings (this may take a minute)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Use FAISS instead of ChromaDB
        print("   💾 Building FAISS vector database...")
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )
        
        # Save to disk for faster loading next time
        try:
            self.vectorstore.save_local("faiss_index")
            print("   ✓ Vector database saved to disk")
        except Exception as e:
            print(f"   ⚠ Could not save to disk: {e}")
        
        print("   🤖 Connecting to QWEN model via Ollama...")
        try:
            llm = OllamaLLM(
                model="qwen2.5:7b",
                temperature=0.2
            )
            # Test connection
            llm.invoke("test")
            print("   ✓ QWEN model connected")
        except Exception as e:
            print(f"   ❌ Error connecting to Ollama: {e}")
            print("   Make sure Ollama is running and qwen2.5:7b is installed")
            raise
        
        prompt_template = """You are an expert content moderator analyzing text for toxicity.

Your task is to classify content as TOXIC, NEUTRAL, or GOOD, then explain your reasoning.

CLASSIFICATION RULES:
- TOXIC: Contains hate speech, threats, harassment, discrimination, personal attacks, obscene language, or any harmful content
- NEUTRAL: Factual statements, disagreements without hostility, questions, constructive criticism
- GOOD: Supportive, encouraging, appreciative, constructive, respectful communication

Use these examples as reference:
{context}

Now analyze this content: "{question}"

Provide your response in this exact format:
Classification: [TOXIC/NEUTRAL/GOOD]
Explanation: [
    Explain why you classified it this way, referencing specific language or tone, 
    then indicate what makes it 
        TOXIC (hate speech, threats, harassment, discrimination, personal attacks, obscene language, or any harmful content) indicate what kind,
        NEUTRAL (Factual statements, disagreements without hostility, questions, constructive criticism) indicate what kind,
        or GOOD (Supportive, encouraging, appreciative, constructive, respectful communication) indicate what kind
]
"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        print("   🔗 Building RAG chain...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def analyze_content(self, content):
        """Analyze content for toxicity using RAG"""
        result = self.qa_chain.invoke({"query": content})
        return result['result']