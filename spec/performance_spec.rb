require "spec_helper"
require "rag_embeddings"
require "benchmark"

RSpec.describe "Performance" do
  let(:text1) { "Performance test one" }
  let(:text2) { "Performance test two" }
  let(:n) { 10_000 }
  let(:embedding_size) { 768 }

  let(:emb1) { Array.new(embedding_size) { rand } }
  let(:emb2) { Array.new(embedding_size) { rand } }

  it "measures the speed of embedding object creation and cosine similarity" do
    creation_time = Benchmark.realtime do
      n.times do
        RagEmbeddings::Embedding.from_array(emb1)
      end
    end

    objs = Array.new(n) { RagEmbeddings::Embedding.from_array(emb1) }
    sim_time = Benchmark.realtime do
      objs.each do |obj|
        obj2 = RagEmbeddings::Embedding.from_array(emb2)
        obj.cosine_similarity(obj2)
      end
    end

    puts "\nEmbedding creation (#{n} times): #{(creation_time * 1000).round} ms"
    puts "Cosine similarity (#{n} times): #{(sim_time * 1000).round} ms"
    puts "RSS: #{(`ps -o rss= -p #{Process.pid}`.to_i / 1024.0).round(2)} MB"

    # Weak expectations, mostly for sanity check
    expect(creation_time).to be < 0.1 # Should be less than 100 milliseconds for 10_000
    expect(sim_time).to be < 0.1      # Should be less than 100 milliseconds for 10_000
  end

  context "memory usage" do
    before { GC.start }
    after  { GC.start }

    it "measures memory usage" do
      before = `ps -o rss= -p #{Process.pid}`.to_i
      objs = Array.new(n) { RagEmbeddings::Embedding.from_array(emb1) }
      after = `ps -o rss= -p #{Process.pid}`.to_i
      puts "\nMemory usage delta: #{((after-before)/1024.0).round(2)} MB for #{n} embeddings"
    end
  end
end
