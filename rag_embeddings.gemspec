require_relative "lib/rag_embeddings/version"

Gem::Specification.new do |spec|
  spec.name          = "rag_embeddings"
  spec.version       = RagEmbeddings::VERSION
  spec.authors       = ["Marco Mastrodonato"]
  spec.email         = ["m.mastrodonato@gmail.com"]

  spec.summary       = "Efficient RAG embedding storage and retrieval"
  spec.description   = "Manage AI vector embeddings in C/Rust with Ruby integration"
  spec.homepage      = "https://rubygems.org/gems/rag_embeddings"
  spec.license       = "MIT"

  spec.files         = Dir["README.md", "LICENSE", "lib/**/*.rb", "ext/**/*.{c,rb,rs,toml}", "Rakefile"]
  spec.extensions    = ["ext/rag_embeddings/extconf.rb"]
  spec.require_paths = ["lib", "ext"]

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = "https://github.com/marcomd/rag_embeddings"

  spec.add_runtime_dependency "sqlite3"
  spec.add_runtime_dependency "langchainrb"
  spec.add_runtime_dependency "faraday"
  spec.add_runtime_dependency "rb_sys", "~> 0.9"

  spec.add_development_dependency "rake-compiler", "~> 1.2"
  spec.add_development_dependency "rake"
  spec.add_development_dependency "rspec"
  spec.add_development_dependency "rubocop"
  spec.add_development_dependency "dotenv"
  spec.add_development_dependency "debug"
end
