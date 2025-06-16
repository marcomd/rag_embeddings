task :compile do
  Dir.chdir("ext/rag_embeddings") do
    # Delete embedding.so or embedding.o
    # Delete embedding.bundle and the folder embedding.bundle.*
    FileUtils.rm_rf(Dir["embedding.so", "embedding.o", "embedding.bundle", "embedding.bundle.*"])
    ruby "extconf.rb"
    system("make")
  end
end
