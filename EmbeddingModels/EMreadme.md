
# Embedding Models using LangChain and Ollama

## 📌 Introduction

Embedding models are machine learning models that convert text into numerical vectors (lists of numbers) that represent the semantic meaning of the text.

They help computers understand relationships between words, sentences, and documents.

Unlike traditional keyword search, embeddings allow us to search based on the meaning of the query rather than just matching exact words.

## 🎯 What I Learned

- What are embedding models?
- How text is converted into numerical vectors.
- Difference between LLMs and embedding models.
- How to generate embeddings using Ollama.
- How to use LangChain's OllamaEmbeddings.
- Difference between embed_query() and embed_documents().
- How semantic similarity works.
- Applications of embeddings in RAG and semantic search.

## ⚙️ Technologies Used

- Python
- LangChain
- Ollama
- NumPy
- Google Colab

## 🧠 How Embedding Models Work

The embedding process:

1. Take text as input.
2. Pass the text to an embedding model.
3. Convert the text into numerical vectors.
4. Store the vectors in a vector database.
5. Compare vectors to find semantically similar content.

Example:

Input:
"I love Machine Learning"

Output (illustrative):

[0.21, 0.45, 0.78, 0.12, ...]

## 📦 Installation

Install the required packages:

```python
!pip install -q langchain-ollama langchain-core numpy
```

Install Ollama and download an embedding model:

```bash
ollama pull nomic-embed-text
```

Make sure the Ollama server is running before executing the Python code.

## 💻 Basic Implementation

```python
from langchain_ollama import OllamaEmbeddings

# Initialize embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Generate embedding for a query
result = embeddings.embed_query(
    "I love learning Machine Learning"
)

print(result)
print(len(result))
```

## 📄 Embedding Multiple Documents

```python
documents = [
    "I love Machine Learning",
    "Artificial Intelligence is interesting",
    "I enjoy learning Python",
    "I like eating pizza"
]

result = embeddings.embed_documents(documents)

for i, vector in enumerate(result):
    print(f"Document {i+1}:")
    print(vector[:5])
```

## 🔍 embed_query() vs embed_documents()

| Method | Purpose |
|---|---|
| embed_query() | Converts one search query into a vector |
| embed_documents() | Converts multiple documents into vectors |

## 📐 Semantic Similarity

Semantic similarity measures how closely two texts are related in meaning.

We can use cosine similarity to compare two embedding vectors.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(
    [result[0]],
    [result[1]]
)

print("Similarity:", similarity[0][0])
```

A higher cosine similarity generally indicates that two vectors point in a more similar direction.

## 🌍 Real-World Applications

1. Semantic Search – Search documents using meaning rather than exact keywords.
2. RAG (Retrieval Augmented Generation) – Retrieve relevant documents before generating answers.
3. Chatbots – Allow chatbots to search through company knowledge bases.
4. Recommendation Systems – Recommend similar products, movies, or articles.
5. Document Classification – Group documents based on their semantic content.
6. Duplicate Detection – Identify documents or sentences with similar meanings.
7. Customer Support – Retrieve relevant answers from previous support conversations.

## 🔗 Embedding Models Available Today

Some currently available models include:

| Model | Provider |
|---|---|
| nomic-embed-text | Ollama |
| embeddinggemma-2 | Ollama |
| qwen3-embedding | Ollama |
| mxbai-embed-large | Ollama |
| bge-m3 | Ollama |
| text-embedding-3-small | OpenAI |
| Gemini Embedding 2 | Google |

## 🔗 Important Concepts

- Vector Embeddings
- Semantic Search
- Cosine Similarity
- Vector Databases
- RAG
- Text Chunking
- Retrieval

## 📚 Conclusion

Through this practice, I learned how embedding models convert text into numerical representations and how these representations help machines understand semantic relationships.

I also explored LangChain's OllamaEmbeddings and learned how embeddings form the foundation of semantic search and RAG-based applications.

This is an important step towards building AI-powered applications and intelligent search systems.

