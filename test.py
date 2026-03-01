from rag_setup import ToxicityRAG, ACTIVE_PROVIDER, LLM_QWEN

print("=" * 50)
print(f"  PROVIDER : {ACTIVE_PROVIDER.value.upper()}")
print(f"  Qwen model  → {LLM_QWEN}")
print("=" * 50)

rag = ToxicityRAG()

# Test Qwen (sarcasm model)
print("\n  [TEST] Qwen / sarcasm model ...")
response = rag.llm_sarcasm.invoke("Reply with only the word: QWEN_OK")
print(f"  Response: {response}")

print("\n" + "=" * 50)
print(f"  ✓ Both models responding via {ACTIVE_PROVIDER.value.upper()}")
print("=" * 50)