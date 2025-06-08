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
      unless query_embedding.is_a?(Array) && query_embedding.all? { |e| e.is_a?(Float) || e.is_a?(Numeric) }
        raise "Query embedding is invalid: #{query_embedding.inspect}"
      end
      query_obj = RagEmbeddings::Embedding.new
      query_obj.set(query_embedding)

      all.map do |id, content, emb|
        unless emb.is_a?(Array) && emb.size == query_embedding.size
          raise "DB embedding for id #{id} is not a valid embedding"
        end
        emb_obj = RagEmbeddings::Embedding.new
        emb_obj.set(emb)
        similarity = emb_obj.cosine_similarity(query_obj)
        [id, content, similarity]
      end.sort_by { |_,_,sim| -sim }.first(k)
    end
  end
end
