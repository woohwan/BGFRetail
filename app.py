#os
import os, tempfile
from pathlib import Path
#front
import streamlit as st
import time
#RAG
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
#Chat model
import boto3
from langchain_community.chat_models import BedrockChat
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#Chat func
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,  MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append(AIMessage("안녕하세요! 주어진 문서에 대해 질문해 주세요."))

st.set_page_config(page_title="RAG_DEMO", page_icon="🦜⛓️")

st.title("_:orange[KAKAO]_ 계정 약관 QA BOT")

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

model_kwargs =  { 
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    #"stop_sequences": ["\n\nHuman"],
}

#Step 1 - load vectorDB
def load_vector_db():
    # load db
    embeddings_model = OpenAIEmbeddings()
    vectorestore = FAISS.load_local('./db/faiss', embeddings_model, allow_dangerous_deserialization=True )
    retriever = vectorestore.as_retriever()
    return retriever


llm = BedrockChat(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

contextualize_q_system_prompt = """Given a chat history and the latest user question \\
which might reference context in the chat history, formulate a standalone question \\
which can be understood without the chat history. Do NOT answer the question, \\
just reformulate it if needed and otherwise return it as is.
"""
contextualize_q_prompt= ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{user_input}"),
    ]
)

contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

# template = """
# You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
# 정답을 모른다면 "카카오 계정 약관에 없는 내용입니다. 다시 질문해주세요."라고 해.
# Question: {question} 
# Context: {context} 
# Answer:
# """
# prompt = ChatPromptTemplate.from_template(template)

retriever = load_vector_db()

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )


# history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, contextualize_q_prompt
# )

#system prompt
qa_system_prompt = """
You are a cafe order assistant. answer consider context and if you don't know said "부정확한 질문입니다. 다시 말씀해주세요.".\
Do not say the information not sure.
Always answer using Korean.
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]

rag_chain = (
    RunnablePassthrough.assign(context=contextualize_q_chain | retriever | format_docs)
    | qa_prompt
    | llm
)


#llm으로 부터 응답 받아오기
def get_response(user_input, chat_history):
    
    response =  rag_chain.invoke({
        "chat_history": chat_history,
        "user_input": user_input
    })
    return response["answer"]

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)   

#user input
user_input = st.chat_input("Your message")
if user_input == '종료':  # 대화 종료 조건
    st.session_state.chat_history.append(AIMessage("대화를 종료합니다"))
    st.stop()
if user_input is not None and user_input != "":
    st.session_state.chat_history.append(HumanMessage(user_input))

    with st.chat_message("Human"):
        st.markdown(user_input)
    
    with st.chat_message("AI"):
        ai_response = get_response(user_input, st.session_state.chat_history)
        st.write(ai_response)

    st.session_state.chat_history.append(AIMessage(ai_response))