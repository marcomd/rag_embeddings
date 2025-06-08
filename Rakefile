task :compile do
  Dir.chdir("ext/rag_embeddings") do
    ruby "extconf.rb"
    system("make")
  end
end
