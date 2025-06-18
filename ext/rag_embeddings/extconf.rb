# frozen_string_literal: true

require 'mkmf'
require "rb_sys/mkmf"

# Check if Rust toolchain is available
def rust_available?
  system('cargo --version > /dev/null 2>&1')
end

# Main build logic
if rust_available?
  puts "🔧 Rust toolchain detected, building Rust extension..."
  create_rust_makefile("rag_embeddings/embedding")
else
  puts "📦 Building C extension..."
  create_makefile("rag_embeddings/embedding")
end