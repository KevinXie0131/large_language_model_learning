# RAG System (rag/main.ipynb)

## Overview
A Retrieval-Augmented Generation (RAG) pipeline built in a Jupyter notebook that answers questions about a Chinese document (doc.md).

## Pipeline Steps
1. **Chunking** — Splits document by double newlines (`\n\n`)
2. **Embedding** — Encodes chunks using `shibing624/text2vec-base-chinese` (SentenceTransformer)
3. **Storage** — Stores embeddings in ChromaDB (ephemeral/in-memory)
4. **Retrieval** — Vector similarity search via ChromaDB, returns top-k chunks
5. **Reranking** — Cross-encoder reranking with `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
6. **Generation** — Sends reranked chunks + query to OpenAI GPT-4o for answer generation

## Tech Stack
- **Embedding model**: `sentence_transformers` (text2vec-base-chinese)
- **Vector store**: `chromadb` (EphemeralClient)
- **Reranker**: `sentence_transformers.CrossEncoder` (mmarco-mMiniLMv2)
- **LLM**: OpenAI API (`gpt-4o`) via `openai` Python SDK
- **Environment**: Designed for Google Colab (uses `google.colab.userdata` for API key)
- **Package manager**: `uv`
