require_relative "lib/rag_embeddings/version"

Gem::Specification.new do |spec|
  spec.name          = "rag_embeddings"
  spec.version       = RagEmbeddings::VERSION
  spec.authors       = ["Marco Mastrodonato"]
  spec.email         = ["m.mastrodonato@gmail.com"]

  spec.summary       = "Efficient RAG embedding storage and retrieval"
  spec.description   = "Manage AI vector embeddings in C with Ruby integration"
  spec.homepage      = "http://example.com"
  spec.license       = "MIT"

  spec.files         = Dir["lib/**/*.rb", "ext/**/*.{c,rb}", "Rakefile"]
  spec.extensions    = ["ext/rag_embeddings/extconf.rb"]
  spec.require_paths = ["lib"]

  spec.add_runtime_dependency "langchainrb"
  spec.add_runtime_dependency "sqlite3"
end
