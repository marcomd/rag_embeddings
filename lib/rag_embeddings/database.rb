require "sqlite3"

module RagEmbeddings
  class Database
    def initialize(path = "embeddings.db")
      @db = SQLite3::Database.new(path)
      @db.execute <<~SQL
        CREATE TABLE IF NOT EXISTS embeddings (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          content TEXT NOT NULL,
          embedding BLOB NOT NULL
        );
      SQL
    end

    def insert(text, embedding)
      blob = embedding.pack("f*")
      @db.execute("INSERT INTO embeddings (content, embedding) VALUES (?, ?)", [text, blob])
    end

    def all
      @db.execute("SELECT id, content, embedding FROM embeddings").map do |id, content, blob|
        [id, content, blob.unpack("f*")]
      end
    end

    # "Raw" search: returns the N texts most similar to the query
    def top_k_similar(query_text, k: 5)
      query_embedding = RagEmbeddings.embed(query_text)
      raise "Wrong embedding size #{query_embedding.size}, #{RagEmbeddings::EMBEDDING_DIMENSION} was expected! Change the configuration." unless query_embedding.size == RagEmbeddings::EMBEDDING_DIMENSION

      query_obj = RagEmbeddings::Embedding.from_array(query_embedding)

      all.map do |id, content, emb|
        raise "Wrong embedding size #{query_embedding.size}, #{RagEmbeddings::EMBEDDING_DIMENSION} was expected! Change the configuration." unless emb.size == RagEmbeddings::EMBEDDING_DIMENSION

        emb_obj = RagEmbeddings::Embedding.from_array(emb)
        similarity = emb_obj.cosine_similarity(query_obj)
        [id, content, similarity]
      end.sort_by { |_,_,sim| -sim }.first(k)
    end
  end
end
