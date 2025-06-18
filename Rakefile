task :compile do
  Dir.chdir("ext/rag_embeddings") do
    # Delete embedding.so or embedding.o
    # Delete embedding.bundle and the folder embedding.bundle.*
    puts "🧹 Cleaning artifacts..."
    system('cargo clean') if File.exist?('Cargo.toml')
    FileUtils.rm_rf(Dir["embedding.so", "embedding.o", "embedding.bundle", "embedding.bundle.*", "target"])
    ruby "extconf.rb"
    system("make")
  end
end
