require "spec_helper"
require "rag_embeddings"
require "benchmark"

RSpec.describe "Performance" do
  let(:text1) { "Performance test one" }
  let(:text2) { "Performance test two" }
  let(:n) { 10_000 }

  let(:emb1) { Array.new(embedding_size) { rand } }
  let(:emb2) { Array.new(embedding_size) { rand } }

  shared_examples "acceptable creation and cosine similarity times" do |options|
    let(:embedding_size) { options[:embedding_size] || raise("embedding_size must be provided") }

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

      puts "\nPerformance test with embedding size: #{embedding_size}"
      puts "Embedding creation (#{n} times): #{(creation_time * 1000).round} ms"
      puts "Cosine similarity (#{n} times): #{(sim_time * 1000).round} ms"
      puts "RSS: #{(`ps -o rss= -p #{Process.pid}`.to_i / 1024.0).round(2)} MB"

      # Weak expectations, mostly for sanity check
      expect(creation_time).to be < 0.5 # Should be less than 100 milliseconds for 10_000
      expect(sim_time).to be < 0.5      # Should be less than 100 milliseconds for 10_000
    end
  end

  shared_examples "a good memory usage" do |options|
    let(:embedding_size) { options[:embedding_size] || raise("embedding_size must be provided") }

    context "memory usage" do
      before { GC.start }
      after  { GC.start }

      it "measures memory usage" do
        puts "\nMemory usage test with embedding size: #{embedding_size}"
        before = `ps -o rss= -p #{Process.pid}`.to_i
        # Create a large number of embedding objects
        objs = Array.new(n) { RagEmbeddings::Embedding.from_array(emb1) }
        after = `ps -o rss= -p #{Process.pid}`.to_i
        puts "Memory usage delta: #{((after-before)/1024.0).round(2)} MB for #{n} embeddings"
      end
    end
  end

  [768, 2048, 3072, 4096].each do |embedding_size|
    sleep 0.1 # Give some time for GC to clean up between tests
    it_behaves_like "acceptable creation and cosine similarity times", { embedding_size: }
    sleep 0.1 # Give some time for GC to clean up between tests
    it_behaves_like "a good memory usage", { embedding_size: }
  end
end
