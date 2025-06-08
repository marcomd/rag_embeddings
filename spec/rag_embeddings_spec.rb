require "spec_helper"
require "rag_embeddings"

RSpec.describe RagEmbeddings do
  let(:text1) { "Hello! This is my first attempt." }
  let(:text2) { "Another completely different sentence." }
  let(:db_path) { "test_embeddings.db" }
  let(:db) { RagEmbeddings::Database.new(db_path) }

  after(:each) { File.delete(db_path) if File.exist?(db_path) }

  it "generates an embedding for text" do
    embedding = RagEmbeddings.embed(text1)
    expect(embedding).to be_a(Array)
    expect(embedding.length).to be > 10 # Should be 768 etc.
    expect(embedding).to all(be_a(Numeric))
  end

  it "creates and sets a C embedding object" do
    embedding = RagEmbeddings.embed(text1)
    obj = RagEmbeddings::Embedding.from_array(embedding)
    expect(obj).to be_a(RagEmbeddings::Embedding)
  end


  it "inserts and reads embeddings in sqlite" do
    emb = RagEmbeddings.embed(text1)
    db.insert(text1, emb)
    results = db.all
    expect(results.size).to eq 1
    id, content, loaded_emb = results.first
    expect(content).to eq(text1)
    expect(loaded_emb).to be_a(Array)
    expect(loaded_emb.size).to eq emb.size
  end

  it "computes cosine similarity between embeddings" do
    emb1 = RagEmbeddings.embed(text1)
    emb2 = RagEmbeddings.embed(text2)
    obj1 = RagEmbeddings::Embedding.from_array(emb1)
    obj2 = RagEmbeddings::Embedding.from_array(emb2)
    sim = obj1.cosine_similarity(obj2)
    expect(sim).to be_a(Float)
    expect(sim).to be <= 1.0
    expect(sim).to be >= -1.0
  end

  it "finds the most similar text for a query" do
    db.insert(text1, RagEmbeddings.embed(text1))
    db.insert(text2, RagEmbeddings.embed(text2))
    # Should find text1 as the most similar to itself
    result = db.top_k_similar(text1, k: 1)
    expect(result).to be_an(Array)
    expect(result.first[1]).to eq(text1)
    expect(result.first[2]).to be_a(Float)
  end
end
