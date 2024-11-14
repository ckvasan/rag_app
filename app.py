from fastapi import FastAPI,Depends,Request
import uuid
import uvicorn
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
from contextlib import asynccontextmanager
from langchain_core.runnables import RunnableLambda,RunnableParallel
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever


base ={}
current_dir = os.path.dirname(os.path.abspath(__file__))

def model():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash")


def load_documents():
    file_path = os.path.join(current_dir,"data","2011_Cricket_World_Cup_final.pdf")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The data in the {file_path} is not present")
    loader = PyPDFLoader(file_path= file_path)
    docs = loader.load()
    docs = docs[:8]
    return docs

def chunk_documents():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap = 200)
    docs = text_splitter.split_documents(load_documents())
    return docs

def get_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings

def vector_store():
    persist_directory = os.path.join(current_dir,"vector_store","world_cup_2011_ds")

    if not os.path.exists(persist_directory):
       docs = chunk_documents()
       db =  Chroma.from_documents(docs,get_embeddings(),collection_name='world_cup_2011_datastore',persist_directory = persist_directory)
    else:
        db = Chroma(collection_name='world_cup_2011_datastore',persist_directory=persist_directory, embedding_function= get_embeddings())
    return db   

def get_retriever():
    db = vector_store()
    return db.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 2, 'fetch_k': 10}
            )

def build_contextual_q_prompt():

    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )
    
    return contextualize_q_prompt

def get_prompt():
    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "Context : "
    "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )
    return qa_prompt
    
async def construct_chain():
    history_aware_retriever = create_history_aware_retriever(
    model(), get_retriever(), build_contextual_q_prompt()
    )
    question_answer_chain = create_stuff_documents_chain(model(), get_prompt())
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    )


    return conversational_rag_chain


@asynccontextmanager
async def prepare_model(app:FastAPI):
    base['chain']= await construct_chain()
    base['session_id'] = str(uuid.uuid4())
    yield
    base.clear()

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]  

app = FastAPI(lifespan=prepare_model)

@app.post("/llm")
def invoke_model(question:str,request:Request):
    return base['chain'].invoke(
    {"input": question},
    config={
        "configurable": {"session_id": base['session_id']}
    }
    )["answer"]

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=5560)
    
