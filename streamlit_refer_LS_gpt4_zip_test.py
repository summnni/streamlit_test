import streamlit as st
import tiktoken
from loguru import logger
import zipfile
from io import BytesIO

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback

def main():
    st.set_page_config(page_title="LS_Chat", page_icon=":owl:")
    st.title("_LS_Data :blue[Chat]_ :owl:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'zip'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in source_documents:
                        st.markdown(doc.metadata['source'], unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response})

def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name

        if '.zip' in file_name:
            with zipfile.ZipFile(BytesIO(doc.getvalue()), 'r') as zip_ref:
                for inner_file_name in zip_ref.namelist():
                    if inner_file_name.endswith(('.pdf', '.docx', '.pptx')):
                        zip_ref.extract(inner_file_name, 'temp')
                        extracted_file_path = f'temp/{inner_file_name}'
                        doc_list.extend(process_document(extracted_file_path))
        else:
            with open(file_name, "wb") as file:
                file.write(doc.getvalue())
                logger.info(f"Uploaded {file_name}")
            doc_list.extend(process_document(file_name))

    return doc_list

def process_document(file_name):
    documents = []
    if '.pdf' in file_name:
        loader = PyPDFLoader(file_name)
    elif '.docx' in file_name:
        loader = Docx2txtLoader(file_name)
    elif '.pptx' in file_name:
        loader = UnstructuredPowerPointLoader(file_name)
    else:
        return documents  # Unsupported file type

    documents = loader.load_and_split()
    return documents

# The rest of the functions (get_text_chunks, get_vectorstore, get_conversation_chain, tiktoken_len) remain unchanged.

if __name__ == '__main__':
    main()
