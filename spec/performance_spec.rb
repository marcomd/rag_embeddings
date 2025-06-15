require "spec_helper"
require "rag_embeddings"
require "benchmark"

RSpec.describe "Performance" do
  let(:text1) { "Performance test one" }
  let(:text2) { "Performance test two" }
  let(:n) { 10_000 }

  let(:emb1) { Array.new(embedding_size) { rand } }
  let(:emb2) { Array.new(embedding_size) { rand } }

  def get_memory_usage
    # Force garbage collection and get memory stats
    GC.start
    stat = GC.stat
    {
      rss_mb: (`ps -o rss= -p #{Process.pid}`.to_i / 1024.0).round(2),
      heap_allocated_pages: stat[:heap_allocated_pages],
      heap_live_slots: stat[:heap_live_slots],
      total_allocated_objects: stat[:total_allocated_objects]
    }
  end

  shared_examples "acceptable creation and cosine similarity times" do |options|
    let(:embedding_size) { options[:embedding_size] || raise("embedding_size must be provided") }

    it "measures the speed of embedding object creation and cosine similarity" do
      # Warm up and clear memory
      3.times { GC.start }

      # Measure creation time without holding references
      creation_time = Benchmark.realtime do
        n.times do
          RagEmbeddings::Embedding.from_array(emb1)
          # Object becomes eligible for GC immediately
        end
      end

      # Create objects for similarity testing
      objs = Array.new(n) { RagEmbeddings::Embedding.from_array(emb1) }

      # Pre-create the comparison embedding to avoid allocation overhead in timing
      emb2_obj = RagEmbeddings::Embedding.from_array(emb2)

      sim_time = Benchmark.realtime do
        objs.each do |obj|
          obj.cosine_similarity(emb2_obj)
        end
      end

      # Clean up before measuring RSS
      objs = nil
      emb2_obj = nil
      GC.start

      rss = (`ps -o rss= -p #{Process.pid}`.to_i / 1024.0).round(2)

      puts "\nPerformance test with embedding size: #{embedding_size}"
      puts "Embedding creation (#{n} times): #{(creation_time * 1000).round} ms"
      puts "Cosine similarity (#{n} times): #{(sim_time * 1000).round} ms"
      puts "RSS after cleanup: #{rss} MB"

      # More reasonable expectations
      expect(creation_time).to be < 1.0, "Creation took #{(creation_time * 1000).round}ms, expected < 1000ms"
      expect(sim_time).to be < 1.0, "Similarity took #{(sim_time * 1000).round}ms, expected < 1000ms"
    end
  end

  shared_examples "a good memory usage" do |options|
    let(:embedding_size) { options[:embedding_size] || raise("embedding_size must be provided") }

    context "memory usage" do
      it "measures memory usage accurately" do
        puts "\nMemory usage test with embedding size: #{embedding_size}"

        # Get baseline memory usage
        baseline = get_memory_usage
        puts "Baseline RSS: #{baseline[:rss_mb]} MB"

        # Create embeddings and measure peak usage
        peak_memory = nil
        objs = []

        # Create in batches to see memory growth pattern
        [2500, 5000, 7500, 10000].each do |batch_size|
          while objs.length < batch_size
            objs << RagEmbeddings::Embedding.from_array(emb1)
          end

          current = get_memory_usage
          peak_memory = current if peak_memory.nil? || current[:rss_mb] > peak_memory[:rss_mb]

          puts "  After #{batch_size} embeddings: #{current[:rss_mb]} MB RSS, " \
                 "#{current[:heap_live_slots]} live objects"
        end

        puts "Peak RSS: #{peak_memory[:rss_mb]} MB"
        puts "Memory delta: #{(peak_memory[:rss_mb] - baseline[:rss_mb]).round(2)} MB"

        # Clear references and force GC
        objs.clear
        objs = nil
        3.times { GC.start }

        # Measure memory after cleanup
        after_gc = get_memory_usage
        puts "After GC RSS: #{after_gc[:rss_mb]} MB"
        puts "Memory retained: #{(after_gc[:rss_mb] - baseline[:rss_mb]).round(2)} MB"

        # Calculate theoretical minimum memory usage
        bytes_per_embedding = 4 * embedding_size + 24  # floats + overhead
        theoretical_mb = (n * bytes_per_embedding) / (1024.0 * 1024.0)
        puts "Theoretical minimum: #{theoretical_mb.round(2)} MB"
        puts "Memory efficiency: #{(theoretical_mb / (peak_memory[:rss_mb] - baseline[:rss_mb]) * 100).round(1)}%"
      end
    end
  end

  shared_examples "memory behavior comparison" do |options|
    let(:embedding_size) { options[:embedding_size] || raise("embedding_size must be provided") }

    it "compares allocation vs deallocation patterns" do
      puts "\nAllocation pattern test with embedding size: #{embedding_size}"

      baseline = get_memory_usage

      # Test 1: Create and immediately discard (tests GC efficiency)
      puts "Test 1: Create and discard immediately"
      start_mem = get_memory_usage

      n.times do |i|
        RagEmbeddings::Embedding.from_array(emb1)
        # Object becomes eligible for GC

        if i % 1000 == 999  # Check every 1000 iterations
          current = get_memory_usage
          puts "  After #{i+1}: #{current[:rss_mb]} MB"
        end
      end

      after_creation = get_memory_usage
      puts "After all creation: #{after_creation[:rss_mb]} MB"

      # Test 2: Hold references then bulk release
      puts "Test 2: Hold references then bulk release"
      refs = Array.new(n) { RagEmbeddings::Embedding.from_array(emb1) }
      with_refs = get_memory_usage
      puts "With all references: #{with_refs[:rss_mb]} MB"

      refs.clear
      refs = nil
      GC.start
      after_release = get_memory_usage
      puts "After release: #{after_release[:rss_mb]} MB"
    end
  end

  [768, 2048, 3072, 4096].each do |embedding_size|
    context "embedding size #{embedding_size}" do
      # Add some separation between tests
      before(:each) { 3.times { GC.start }; sleep 0.1 }

      it_behaves_like "acceptable creation and cosine similarity times", { embedding_size: }
      it_behaves_like "a good memory usage", { embedding_size: }
      it_behaves_like "memory behavior comparison", { embedding_size: }
    end
  end
end
