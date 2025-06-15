# 💎 Rag Embeddings  

[![Gem Version](https://badge.fury.io/rb/rag_embeddings.svg?icon=si%3Arubygems)](https://badge.fury.io/rb/rag_embeddings)

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

## 🌍 Real-world Use Cases

- **Question Answering over Documents:** Instantly search and retrieve the most relevant document snippets from thousands of articles, FAQs, or customer support logs in your Ruby app.
- **Semantic Search for E-commerce:** Power product search with semantic understanding, returning items similar in meaning, not just keywords.
- **Personalized Recommendations:** Find related content (articles, products, videos) by comparing user preferences and content embeddings.
- **Knowledge Base Augmentation:** Use with OpenAI or Ollama to enhance chatbots, letting them ground answers in your company’s internal documentation or wiki.
- **Fast Prototyping for AI Products:** Effortlessly build MVPs for RAG-enabled chatbots, semantic search tools, and AI-driven discovery apps—all in native Ruby.

---

## 👷 Requirements

- Ruby >= 3.3
- `langchainrb` (for embedding)
- At the moment `ollama` is used as LLM so it must be active and working, although there are some workarounds
- `sqlite3` (for storage)

## 🔧 Installation

Requires a working C compiler in order to build the native extension

`gem install rag_embeddings`

If you'd rather install it using bundler, add a line for it in your Gemfile (but set the require option to false, as it is a standalone tool):

```ruby
gem "rag_embeddings", require: false
```


## 🧪 Practical examples

### 1. Generate an embedding from text

```ruby
require "rag_embeddings"
embedding = RagEmbeddings.embed("Hello world, this is RAG!")
# embedding is a float array
```

The default model is llama3.2 but you can set another one (reload the console as the llm is memoized):

```ruby
embedding = RagEmbeddings.embed("Hello world, this is RAG!", model: 'qwen3:0.6b')
````

### 2. Create a C embedding object

```ruby
c_embedding = RagEmbeddings::Embedding.from_array(embedding)
puts "Dimension: #{c_embedding.dim}"
# Dimension: 1024 # qwen3:0.6b
# Dimension: 3072 # llama3.2

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

### 5. Batch-index a folder of documents

```ruby
# load all .txt files
files = Dir["./docs/*.txt"].map { |f| [File.basename(f), File.read(f)] }

db = RagEmbeddings::Database.new("knowledge_base.db")
files.each do |name, text|
  vector = RagEmbeddings.embed(text)
  db.insert(name, vector)
end

puts "Indexed #{files.size} documents."
```

### 6. Simple Retrieval-Augmented Generation (RAG) loop

```ruby
require "openai"        # or your favorite LLM client

# 1) build or open your vector store
db = RagEmbeddings::Database.new("knowledge_base.db")

# 2) embed your user question
client      = OpenAI::Client.new(api_key: ENV.fetch("OPENAI_API_KEY"))
q_embedding = client.embeddings(
  parameters: {
    model: "text-embedding-ada-002",
    input: "What are the benefits of retrieval-augmented generation?"
  }
).dig("data", 0, "embedding")

# 3) retrieve top-3 relevant passages
results = db.top_k_similar(q_embedding, k: 3)

# 4) build a prompt for your LLM
context = results.map { |id, text, score| text }.join("\n\n---\n\n")
prompt  = <<~PROMPT
  You are an expert.  
  Use the following context to answer the question:

  CONTEXT:
  #{context}

  QUESTION:
  What are the benefits of retrieval-augmented generation?
PROMPT

# 5) call the LLM for final answer
response = client.chat(
  parameters: {
    model: "gpt-4o",
    messages: [{ role: "user", content: prompt }]
  }
)
puts response.dig("choices", 0, "message", "content")

```

### 7. In-memory store for fast prototyping

```ruby
# use SQLite :memory: for ephemeral experiments
db = RagEmbeddings::Database.new(":memory:")

# insert & search exactly as with a file-backed DB
db.insert("Quick test", RagEmbeddings.embed("Quick test"))
db.top_k_similar("Test", k: 1)
```

---

## 🏗️ How it works

**rag_embeddings** combines the simplicity of Ruby with the performance of C to deliver fast vector operations for RAG applications.

### Architecture Overview

The library uses a **hybrid memory-storage approach**:

1. **In-Memory Processing**: All vector operations (cosine similarity calculations, embedding manipulations) happen entirely in memory using optimized C code
2. **Persistent Storage**: SQLite serves as a simple, portable storage layer for embeddings and associated text
3. **Dynamic C Objects**: Embeddings are managed as native C structures with automatic memory management

### Key Components

**C Extension (`embedding.c`)**
- Handles all computationally intensive operations
- Manages dynamic vector dimensions (adapts to any LLM output size)
- Performs cosine similarity calculations with optimized algorithms
- Ensures memory-safe operations with proper garbage collection integration

**Ruby Interface**
- Provides an intuitive API for vector operations
- Integrates seamlessly with LLM providers via langchainrb
- Handles database operations and query orchestration

**SQLite Storage**
- Stores embeddings as BLOBs alongside their associated text
- Provides persistent storage without requiring external databases
- Supports both file-based and in-memory (`:memory:`) databases
- Enables portable, self-contained applications

### Processing Flow

1. **Text → Embedding**: Generate vectors using your preferred LLM (Ollama, OpenAI, etc.)
2. **Memory Allocation**: Create C embedding objects with `Embedding.from_array()`
3. **Storage**: Persist embeddings and text to SQLite for later retrieval
4. **Query Processing**:
    - Load query embedding into memory
    - Compare against stored embeddings using fast C-based cosine similarity
    - Return top-K most similar results ranked by similarity score

### Why This Design?

**Performance**: Critical operations run in optimized C code, delivering significant speed improvements over pure Ruby implementations.

**Memory Efficiency**: While embeddings are stored in SQLite, all vector computations happen in memory, avoiding I/O bottlenecks during similarity calculations.

**Simplicity**: SQLite eliminates the need for complex vector database setups while maintaining good performance for moderate-scale applications.

**Portability**: The entire knowledge base fits in a single SQLite file, making deployment and backup trivial.

### Performance Characteristics

- **Embedding creation**: ~82ms for 10,000 operations
- **Cosine similarity**: ~107ms for 10,000 calculations
- **Memory usage**: ~34MB for 10,000 embeddings
- **Scalability**: Suitable for thousands to tens of thousands of vectors

For applications requiring millions of vectors, consider specialized vector databases (Faiss, sqlite-vss) while using this library for prototyping and smaller-scale production use.

## 🎛️ Customization

- Embedding provider: switch model/provider in engine.rb (Ollama, OpenAI, etc)
- Database: set the SQLite file path as desired

If you need to customize the c part (`ext/rag_embeddings/embedding.c`), recompile it with:

`rake compile`

---

## 🏁 Running the test suite

To run all specs (RSpec required):

`bundle exec rspec`

## ⚡️ Performance

`bundle exec rspec spec/performance_spec.rb`

You'll get something like this in random order:

```bash
Performance test with embedding size: 768
Embedding creation (10000 times): 19 ms
Cosine similarity (10000 times): 27 ms
RSS: 132.3 MB

Memory usage test with embedding size: 768
Memory usage delta: 3.72 MB for 10000 embeddings


Performance test with embedding size: 2048
Embedding creation (10000 times): 69 ms
Cosine similarity (10000 times): 73 ms
RSS: 170.08 MB

Memory usage test with embedding size: 2048
Memory usage delta: 25.11 MB for 10000 embeddings


Performance test with embedding size: 3072
Embedding creation (10000 times): 98 ms
Cosine similarity (10000 times): 112 ms
RSS: 232.97 MB

Memory usage test with embedding size: 3072
Memory usage delta: 60.5 MB for 10000 embeddings


Performance test with embedding size: 4096
Embedding creation (10000 times): 96 ms
Cosine similarity (10000 times): 140 ms
RSS: 275.2 MB

Memory usage test with embedding size: 4096
Memory usage delta: 92.41 MB for 10000 embeddings
```

## 📬 Contact & Issues
Open an issue or contact the maintainer for questions, suggestions, or bugs.


Happy RAG! 🚀