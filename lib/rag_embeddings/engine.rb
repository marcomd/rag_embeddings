require "langchainrb"

module RagEmbeddings
  DEFAULT_MODEL = "llama3.2".freeze

  def self.llm(model: DEFAULT_MODEL)
    @llm ||= Langchain::LLM::Ollama.new(url: "http://localhost:11434",
                                        default_options: {
                                          temperature: 0.1,
                                          chat_model: model,
                                          completion_model: model,
                                          embedding_model: model,
                                        }
    )
  end

  def self.embed(text, model: DEFAULT_MODEL)
    llm(model:).embed(text:).embedding
  end
end
