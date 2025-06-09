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

## 🔧 Installation

Requires a working C compiler!

`gem install rag_embeddings`

Or add to your Gemfile:

```ruby
gem "rag_embeddings"
```

bundle install


## 🧪 Practical examples

### 1. Generate an embedding from text

```ruby
text = "Hello world, this is RAG!"
embedding = RagEmbeddings.embed(text)
# embedding is a float array
```

The default model is llama3.2 but you can set another one (reload the console as the llm is memoized):

```ruby
embedding = RagEmbeddings.embed(text, model: 'qwen3:0.6b')
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

- Embeddings are managed as dynamic C objects for efficiency (variable dimension).
- The only correct way to construct an embedding object is using .from_array.
- Langchainrb integration lets you easily change the embedding provider (Ollama, OpenAI, etc).
- Storage uses local SQLite with embeddings as BLOB, for maximum portability and simplicity.

## 🎛️ Customization

- Embedding provider: switch model/provider in engine.rb (Ollama, OpenAI, etc)
- Database: set the SQLite file path as desired

If you need to customize the c part (`ext/rag_embeddings/embedding.c`), recompile it with:

`rake compile`

## 🔢 Embeddings dimension

The size of embeddings is dynamic and fits with what the LLM provides.

## 👷 Requirements

- Ruby >= 3.3
- langchainrb (for embedding)
- sqlite3 (for storage)
- A working C compiler

## 📑 Notes

- Always create embeddings with .from_array
- All memory management is idiomatic and safe
- For millions of vectors, consider vector DBs (Faiss, sqlite-vss, etc.)

---

## 🏁 Running the test suite

To run all specs (RSpec required):

`bundle exec rspec`

## ⚡️ Performance

`bundle exec rspec spec/performance_spec.rb`

```bash
Embedding creation (10000 times): 82 ms
Cosine similarity (10000 times): 107 ms
RSS: 186.7 MB
.
Memory usage delta: 33.97 MB for 10000 embeddings
.

Finished in 0.42577 seconds (files took 0.06832 seconds to load)
2 examples, 0 failures
```

## 📬 Contact & Issues
Open an issue or contact the maintainer for questions, suggestions, or bugs.


Happy RAG! 🚀