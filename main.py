
# Import
import streamlit as st
import time

# RAG
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

# Prompt
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts.chat import ChatPromptTemplate

# Claude
import boto3
from langchain_aws import ChatBedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Chat history
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Streamlit
st.set_page_config(page_title="RAG_DEMO", page_icon="🦜⛓️")
st.title("_:orange[KAKAO]_ 계정 약관 QA BOT")

# Get Retriever
def load_vector_db():
    # load db
    embeddings_model = OpenAIEmbeddings()
    vectorestore = FAISS.load_local('./db/faiss', embeddings_model, allow_dangerous_deserialization=True )
    retriever = vectorestore.as_retriever()
    return retriever

retriever = load_vector_db()

# Get LLM
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

model_kwargs =  { 
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

llm = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Get Retriever
def load_vector_db():
    # load db
    embeddings_model = OpenAIEmbeddings()
    vectorestore = FAISS.load_local('./db/faiss', embeddings_model, allow_dangerous_deserialization=True )
    retriever = vectorestore.as_retriever()
    return retriever

retriever = load_vector_db()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prompting for RAG
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise.
            \n\n
            {context}
            Question: {input}"""
        ),
    ]
)

# chat history
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Set up memory
# msgs = StreamlitChatMessageHistory(key="chat_history")
# if len(msgs.messages) == 0:
#     msgs.add_ai_message("안녕하세요! 주어진 문서에 대해 질문해 주세요.")

# view_messages = st.expander("View the message contents in session state")

# chain_with_history = RunnableWithMessageHistory(
#     rag_chain,
#     lambda session_id: msgs,  # Always return the instance created earlier
#     input_messages_key="input",
#     history_messages_key="chat_history",
# )

# config = {"configurable": {"session_id": "any"}}
# chat history 초기화
# chat_history = []
# chat history 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up"):
    # st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # ai_msg_1 = chain_with_history.invoke({"input": prompt, "chat_history": msgs}, config)
    ai_msg_1 = rag_chain.invoke({"input": prompt, "chat_history": st.session_state.messages})
    st.chat_message("assistant").write(ai_msg_1["answer"])
    # st.session_state.messages.extend([HumanMessage(content=prompt), AIMessage(ai_msg_1["answer"])])
    st.session_state.messages.append({"role": "assistant", "content": ai_msg_1["answer"]})