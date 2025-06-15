task :compile do
  Dir.chdir("ext/rag_embeddings") do
    ruby "extconf.rb"
    system("make")
  end
end

task :recompile do
  # Delete ext/rag_embeddings/embedding.so if it exists
  # Delete ext/rag_embeddings/embedding.o if it exists
  # Delete ext/rag_embeddings/embedding.bundle if it exists
  # Delete folder ext/rag_embeddings/embedding.bundle.* if it exists
  FileUtils.rm_rf(Dir["ext/rag_embeddings/embedding.o*", "ext/rag_embeddings/embedding.bundle.*"])
  Rake::Task["compile"].invoke
end
