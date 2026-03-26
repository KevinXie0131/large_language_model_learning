from typing import List

import chromadb
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

# ========== 1. 文档分块 Chunking ==========
# 读取文档并按双换行符将其分割成多个文本块
# Read the document and split it into chunks by double newlines

def split_into_chunks(doc_file: str) -> List[str]:
    with open(doc_file, 'r') as file:
        content = file.read()
    return [chunk for chunk in content.split("\n\n")]


chunks = split_into_chunks("doc.md")

for i, chunk in enumerate(chunks):
    print(f"[{i}] {chunk}\n")

# ========== 2. 向量嵌入 Embedding ==========
# 加载中文句向量模型，将每个文本块编码为向量
# Load a Chinese sentence embedding model and encode each chunk into a vector

embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")


def embed_chunk(chunk: str) -> List[float]:
    # normalize_embeddings=True 确保向量为单位长度，适用于余弦相似度计算
    # normalize_embeddings=True ensures vectors have unit length for cosine similarity
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()


embedding = embed_chunk("测试内容")
print(len(embedding))
print(embedding)

# 对所有文本块进行向量嵌入
# Embed all chunks
embeddings = [embed_chunk(chunk) for chunk in chunks]

print(len(embeddings))
print(embeddings[0])

# ========== 3. 向量存储 Storage ==========
# 将文本块及其向量存储到内存中的 ChromaDB 集合
# Store chunk texts and their embeddings in an in-memory ChromaDB collection

chromadb_client = chromadb.EphemeralClient()
chromadb_collection = chromadb_client.get_or_create_collection(name="default")


def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chromadb_collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(i)]  # 使用文本块索引作为唯一ID / Use chunk index as unique ID
        )


save_embeddings(chunks, embeddings)

# ========== 4. 检索 Retrieval ==========
# 将查询语句转为向量，从 ChromaDB 中检索最相似的文本块
# Embed the query and retrieve the most similar chunks from ChromaDB


def retrieve(query: str, top_k: int) -> List[str]:
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    # results['documents'] 是一个嵌套列表；[0] 取第一个（也是唯一的）查询的结果
    # results['documents'] is a nested list; [0] gets the first (and only) query's results
    return results['documents'][0]


query = "哆啦A梦使用的3个秘密道具分别是什么？"
retrieved_chunks = retrieve(query, 5)

for i, chunk in enumerate(retrieved_chunks):
    print(f"[{i}] {chunk}\n")

# ========== 5. 重排序 Reranking ==========
# 使用交叉编码器根据与查询的相关性对检索到的文本块进行重排序
# Use a cross-encoder to rerank retrieved chunks by relevance to the query


def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    # 构建（查询, 文本块）配对，供交叉编码器评分
    # Create (query, chunk) pairs for the cross-encoder to score
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    # 按分数降序排列，返回前 top_k 个文本块
    # Sort chunks by score in descending order and return the top_k
    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scored_chunks][:top_k]


reranked_chunks = rerank(query, retrieved_chunks, 3)

for i, chunk in enumerate(reranked_chunks):
    print(f"[{i}] {chunk}\n")

# ========== 6. 生成回答 Generation ==========
# 将查询和重排序后的上下文文本块发送给 OpenAI GPT-4o 生成回答
# Send the query and reranked context chunks to OpenAI GPT-4o for answer generation

# 从 .env 文件加载 API 密钥（替代 Google Colab 的 userdata 方式）
# Load API key from .env file (replaces google.colab.userdata for local use)
load_dotenv()
openai_client = openai.OpenAI()


def generate(query: str, chunks: List[str]) -> str:
    prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。

用户问题: {query}

相关片段:
{"\n\n".join(chunks)}

请基于上述内容作答，不要编造信息。"""

    print(f"{prompt}\n\n---\n")

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


answer = generate(query, reranked_chunks)
print(answer)
