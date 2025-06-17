EXTENSION_PATH = "ext/rag_embeddings".freeze

task :compile do
  Dir.chdir(EXTENSION_PATH) do
    ruby "build.rb"
    # system("make")
  end
end

task :clean do
  Dir.chdir(EXTENSION_PATH) do
    ruby "build.rb clean"
    # system("make")
  end
end
