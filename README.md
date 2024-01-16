# youtube_Assistant-Transcription

This is a `LangChain` based project which is actually an assistant to users. The project performs the following taks:

-- It asks the users to upload a youtube URL.
-- Then it asks the user to ask any question about the video

The model transcripts the video and creates embeddings which are stored in `faiss_index`. Using similarity search, model answers to the user's questions. If the question was mentioned or discussed in the video, model will give its response. But, if the model doesn't have enough information regarding the question then it will say "I don't know".
