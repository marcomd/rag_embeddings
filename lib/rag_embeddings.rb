require_relative "rag_embeddings/version"
require_relative "rag_embeddings/engine"
require_relative "rag_embeddings/database"

# Loads the compiled C extension
require "rag_embeddings/embedding"

require "faraday"
