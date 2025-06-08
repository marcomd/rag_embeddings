# 💎 Rag Embeddings  

[![Gem Version](https://badge.fury.io/rb/rag_embeddings.svg)](https://badge.fury.io/rb/rag_embeddings)

**rag_embeddings** is a native Ruby library for efficient storage and comparison of AI-generated embedding vectors (float arrays) using high-performance C extensions. It is designed for seamless integration with external LLMs (Ollama, OpenAI, Mistral, etc) and works perfectly for RAG (Retrieval-Augmented Generation) applications.

- **C extension for maximum speed** in cosine similarity and vector allocation
- **Compatible with langchainrb** for embedding generation (Ollama, OpenAI, etc)
- **SQLite-based storage** with vector search capabilities
- **RSpec tested**

---

## 📦 Features

- Creation of embedding objects from LLM-generated float arrays
- Cosine similarity calculation in C for speed and safety
- Embedding + text storage in SQLite (BLOB)
- Retrieve top-K most similar texts to a query using cosine similarity
- Memory-safe and 100% Ruby compatible
- Plug-and-play for RAG, semantic search, and retrieval AI

---

## 🔧 Installation

Add to your Gemfile:

```ruby
gem "rag_embeddings"
gem "langchainrb"
gem "faraday"
gem "sqlite3"
```

bundle install
rake compile

(Requires a working C compiler!)

## 🏁 Running the test suite

To run all specs (RSpec required):

`bundle exec rspec`

## 🧪 Practical examples

### 1. Generate an embedding from text

```ruby
require "rag_embeddings"

text = "Hello world, this is RAG!"
embedding = RagEmbeddings.embed(text)
# embedding is a float array
```

### 2. Create a C embedding object

```ruby
c_embedding = RagEmbeddings::Embedding.from_array(embedding)
puts "Dimension: #{c_embedding.dim}"
puts "Ruby array: #{c_embedding.to_a.inspect}"
```

### 3. Compute similarity between two texts

```ruby
emb1 = RagEmbeddings.embed("Hello world!")
emb2 = RagEmbeddings.embed("Hi universe!")
obj1 = RagEmbeddings::Embedding.from_array(emb1)
obj2 = RagEmbeddings::Embedding.from_array(emb2)
sim = obj1.cosine_similarity(obj2)
puts "Cosine similarity: #{sim}"  # Value between -1 and 1
```

### 4. Store and search embeddings in a database

```ruby
db = RagEmbeddings::Database.new("embeddings.db")
db.insert("Hello world!", RagEmbeddings.embed("Hello world!"))
db.insert("Completely different sentence", RagEmbeddings.embed("Completely different sentence"))

# Find the most similar text to a query
result = db.top_k_similar("Hello!", k: 1)
puts "Most similar text: #{result.first[1]}, score: #{result.first[2]}"
```

## 🏗️ How it works

- Embeddings are managed as dynamic C objects for efficiency (variable dimension).
- The only correct way to construct an embedding object is using .from_array.
- Langchainrb integration lets you easily change the embedding provider (Ollama, OpenAI, etc).
- Storage uses local SQLite with embeddings as BLOB, for maximum portability and simplicity.

## 🎛️ Customization

- Embedding provider: switch model/provider in engine.rb (Ollama, OpenAI, etc)
- Database: set the SQLite file path as desired

## 👷 Requirements

- Ruby >= 3.3
- langchainrb (for embedding)
- sqlite3 (for storage)
- A working C compiler

## 📑 Notes

- Always create embeddings with .from_array
- All memory management is idiomatic and safe
- For millions of vectors, consider vector DBs (Faiss, sqlite-vss, etc.)

## 📬 Contact & Issues
Open an issue or contact the maintainer for questions, suggestions, or bugs.


Happy RAG! 🚀