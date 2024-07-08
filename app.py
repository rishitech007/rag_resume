import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from utils.constants import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, GPT_MODEL, FAISS_DB, TEMPERATURE, PDF_DATA
import os



def pdf_read(pdf_doc):
    """
    Method to read PDF files and extract text from it.
    
    Parameters:
    pdf_doc(list): List of PDF files.
    
    Returns:
    (str): String containing the data in the pdf file.
    """
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    """
    Method to extract chunks from raw text.
    
    Parameters:
    text(str): String of the read text file.
    
    Returns:
    chunks(list): Chunks of text based on splitter method used.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_text(text)
    return chunks


def vector_store(text_chunks):    
    """
    Method to create a vector store of the embeddings.

    Parameters:
    text_chunks(list): List of chunks of text from raw strings.
    Returns:
    None
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_DB)


def get_conversational_chain(tools, ques):
    llm = ChatOpenAI(model_name=GPT_MODEL, temperature=TEMPERATURE)
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant about Hrishikesh Rao. He is a data scientist with 8 years of experience in retail and security with a PhD and Masters from Georgia Tech. 
            Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
    tool=[tools]
    agent = create_tool_calling_agent(llm, tool, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    response=agent_executor.invoke({"input": ques})
    print(response)
    st.write("Reply: ", response['output'])



def user_input(user_question):
    """
    Method to call RAG chain to get response for query from user.

    Parameters:
    user_question(str): Query asked by user
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    new_db = FAISS.load_local(FAISS_DB, embeddings,allow_dangerous_deserialization=True)
    retriever=new_db.as_retriever()
    retrieval_chain= create_retriever_tool(retriever, "pdf_extractor", "This tool is to give answer to queries from the pdf")
    get_conversational_chain(retrieval_chain, user_question)





def main():
    st.set_page_config("Chat about Hrishikesh's profile")
    st.header("RAG based Chat about Hrishikesh Rao")

    user_question = st.text_input("Ask a Question about Hrishikesh")
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
 
    if not openai_api_key:
        st.info('Please add your OpenAI API key')
        st.stop()
    else:
        os.environ['OPENAI_API_KEY']=openai_api_key
    
    if user_question and openai_api_key:
        user_input(user_question)
    raw_text = pdf_read(PDF_DATA)
    text_chunks = get_chunks(raw_text)
    vector_store(text_chunks)

if __name__ == "__main__":
    main()
