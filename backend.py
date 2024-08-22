import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Check if the API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

def load_and_process_document(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

def create_vector_store(texts):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings)
    return vectorstore

def setup_qa_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.7),
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"output_key": "answer"}
    )
    
    return qa_chain

# Global variable to store the QA chain
qa_chain = None

def initialize_backend():
    global qa_chain
    try:
        texts = load_and_process_document("data/faid.pdf")
        vectorstore = create_vector_store(texts)
        qa_chain = setup_qa_chain(vectorstore)
        print("Backend setup completed successfully.")
    except Exception as e:
        print(f"An error occurred during setup: {str(e)}")
        raise

def process_query(query):
    global qa_chain
    if qa_chain is None:
        initialize_backend()
    try:
        result = qa_chain({"question": query})
        return result["answer"]
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    initialize_backend()
    test_query = "What is this document about?"
    print(f"Test Query: {test_query}")
    print(f"Response: {process_query(test_query)}")