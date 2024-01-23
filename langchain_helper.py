from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.document_loaders.youtube import YoutubeLoader
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

embeddings = OpenAIEmbeddings()

def create_vector_db_from_youtube_url(video_url: str):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    chunk = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = chunk.split_documents(transcript)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")

def return_db_embeddings()-> FAISS:
    new_db = FAISS.load_local("faiss_index", embeddings)
    return new_db


def get_response_from_query(new_db, query, k=4):
    docs = new_db.similarity_search(query, k)
    docs_page_content = ''.join([d.page_content for d in docs])


    llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0.6)
    prompt = PromptTemplate(input_variables=['question', 'docs'], 
        template="""
            You are a helpful assisstant that can answer questions about video bases on that 
            video transcript.

            Answer the following question: {question}
            By searching the following video transcript: {docs}

            Only use the fractual information from the docs to answer the qesution.

            If you feel like you don't know have enough information to answer, just say "I don't know".
            Your answers should be detailed.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt, output_key='query-text')
    respose = chain.invoke({'question': query, 'docs': docs_page_content })
    return respose['query-text']


video_url = 'https://youtu.be/TRjq7t2Ms5I?si=PhZNe2Ys09CqBRtJ'
if __name__ == '__main__':
    new_db = return_db_embeddings()
    print(get_response_from_query(new_db, 'What are RAGS ?'))

