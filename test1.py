import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import tiktoken
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI
import os
import logging

# Streamlit 앱 설정
st.set_page_config(page_title="PDF 기반 챗봇", page_icon=":books:")
st.title("PDF 기반 챗봇")

# PDF 파일 경로 리스트 정의
pdf_files = [
    "C:\Users\\USER\\Downloads\\chatbot\\미리보는 CES 2024 All Together, All On_vf.pdf",
    "C:\\Users\\USER\\Downloads\\chatbot\\CES 2024로 본 미래 산업 트렌드.pdf",
    # 추가적인 파일 경로들...
]

@st.cache
def load_and_process_documents(pdf_files):
    all_pages = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pages = loader.load_and_split()
        all_pages.extend(pages)

    tokenizer = tiktoken.get_encoding("cl100k_base")
    def tiktoken_len(text):
        tokens=tokenizer.encode(text)
        return len(tokens)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function=tiktoken_len)
    docs = text_splitter.split_documents(all_pages)
    return docs

docs = load_and_process_documents(pdf_files)

# 임베딩 설정 및 벡터 데이터베이스 구축
@st.cache(allow_output_mutation=True)
def build_vector_db(docs):
    model_name = 'jhgan/ko-sbert-nli' # 한국어 모델
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    db = FAISS.from_documents(docs, embeddings)
    return db

db = build_vector_db(docs)

# 사용자 질문 입력
user_query = st.text_input("질문을 입력해주세요.", "")

# OpenAI API 키 환경변수에서 불러오기
openai_api_key = os.getenv("OPENAI_API_KEY")

# 질문 처리 및 응답
if user_query and openai_api_key:
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(), llm=llm
    )
    
    unique_docs = retriever_from_llm.get_relevant_documents(query=user_query)

    # 결과 출력
    st.write(f"질문: {user_query}\n")
    for i, doc in enumerate(unique_docs):
        st.write(f"{i+1}번째 유사 문서 유사도: {round(doc.score, 2)}")
        st.text_area(label=f"문서 {i+1}", value=doc.page_content, height=150)
