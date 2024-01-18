!pip install Streamlit
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Streamlit 앱 타이틀 설정
st.title("CES 정보 챗봇")

# PDF 파일 경로 리스트 정의
pdf_files = [
    "/content/drive/MyDrive/content/미리보는 CES 2024 All Together, All On_vf.pdf",
    "/content/drive/MyDrive/content/CES 2024로 본 미래 산업 트렌드.pdf",
    # 추가적인 파일 경로들...
]

# 모든 PDF 파일들의 페이지를 담을 리스트
all_pages = []

# 각 PDF 파일 로드 및 페이지 분할
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()
    all_pages.extend(pages)

# 모든 페이지를 분할하기 위한 텍스트 스플리터 설정
# 여기서 tiktoken_len은 정의되어 있어야 함
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function=tiktoken_len)
docs = text_splitter.split_documents(all_pages)

# 임베딩 설정
model_name = 'jhgan/ko-sbert-nli'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
ko = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

# FAISS 인덱스 생성
db = FAISS.from_documents(docs, ko)

# Streamlit 입력 필드 및 버튼
user_query = st.text_input("질문을 입력하세요:")

if st.button("검색"):
    if user_query:
        # 사용자 질문 인코딩
        query_embedding = ko.encode([user_query])

        # FAISS 인덱스를 사용하여 유사한 문서 검색
        similar_docs = db.search(query_embedding, k=3)

        # 결과 출력
        for i, doc in enumerate(similar_docs):
            st.write(f"#{i+1} 유사 문서 유사도: {round(doc[1], 2)}")
            st.write(doc[0].page_content)
            st.write("-----")
    else:
        st.write("질문을 입력하세요.")
