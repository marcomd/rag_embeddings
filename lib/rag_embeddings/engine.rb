require "langchainrb"

module RagEmbeddings
  MODEL = "gemma3".freeze

  def self.llm
    @llm ||= Langchain::LLM::Ollama.new(url: "http://localhost:11434", default_options: { temperature: 0.1, model: MODEL })
  end

  def self.embed(text)
    llm.embed(text: text).embedding
  end
end
