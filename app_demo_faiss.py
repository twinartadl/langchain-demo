import streamlit as st
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory, StreamlitChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
import time
from typing import List
from langchain_community.chat_message_histories import (
    PostgresChatMessageHistory,
)
from langchain_core.documents import Document
from langchain_core.runnables import chain
import warnings
# Suppress specific warning
warnings.filterwarnings("ignore", category=DeprecationWarning)


default_session_id = str(time.time())

# Check if session_id is already in session_state
if 'session_id' not in st.session_state:
    st.session_state.session_id = default_session_id
# Create a sidebar for input
with st.sidebar:
    input_session_id = st.text_input("Insert Session ID", st.session_state.session_id)
    if input_session_id:
        st.session_state.session_id = input_session_id
# Use the session_id from session_state
session_id = st.session_state.session_id
# Print the session ID
st.write("Session ID: ", session_id)





postgres_history = PostgresChatMessageHistory(
    connection_string="postgresql://root:root@localhost:5433/langchain_demo_faiss",
    session_id=session_id,
)

# Set up Azure OpenAI credentials
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://cog-kguqugfu5p2ki.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "4657af893faf48e5bd81208d9f87f271"

deployment_name = "4o"

with st.sidebar:
    st.subheader("Session ID:", divider='rainbow')
    st.text(f"{session_id}")

with st.sidebar:
    # Add deployment name selection
    deployment_name = st.selectbox(
        "Select deployment name:",
        options=["gpt-4o", "chat4", "chat16k"],
        index=0  # Default to "gpt-4o"
    )


llm = AzureChatOpenAI(deployment_name=deployment_name, temperature=0)

# Initialize Azure OpenAI embedding and LLM
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="embedding",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=os.environ["OPENAI_API_VERSION"]
)

st.title("Deeeplabs AI Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

#Load History
for postgres_history.message in postgres_history.messages:
    if postgres_history.message.type=='human':
        st.session_state.messages.append({"role": "user", "content": postgres_history.message.content})
    elif postgres_history.message.type=='ai':
        st.session_state.messages.append({"role": "assistant", "content": postgres_history.message.content})

if "vector_store" not in st.session_state:
    # Try to load existing vector store
    if os.path.exists("./faiss_index"):
        st.session_state.vector_store = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        st.session_state.vector_store = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Create a directory to store uploaded files
UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = StreamlitChatMessageHistory("deeeplabs")
    return store[session_id]

# Function to list files in the upload directory
def list_uploaded_files():
    return [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]

def remove_all_files():
    # Remove all files from the upload directory
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    
    if st.session_state.vector_store is not None:
        st.session_state.vector_store = FAISS.from_texts([""], embeddings)
        st.session_state.vector_store.save_local("./faiss_index")
    
    # Reset uploaded files in session state
    st.session_state.uploaded_files = []
    
    st.success("Knowledgebase has been reset")

    st.experimental_rerun()

# Update the list of uploaded files
st.session_state.uploaded_files = list_uploaded_files()

with st.sidebar:
    # Display the list of uploaded files
    st.subheader("Knowledgebase:")
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.session_state.uploaded_files:
            for file in st.session_state.uploaded_files:
                st.write(f"- {file}")
        else:
            st.write("No files uploaded yet.")

    with col2:
        if st.button("Remove All Files"):
            remove_all_files()
            st.experimental_rerun()

uploaded_file = None

with st.sidebar:
    # File uploader
    uploaded_file = st.file_uploader("Upload to knowledgebase (TXT, PDF, or DOCX)", type=["txt", "pdf", "docx"])

if uploaded_file:
    # Check if the file is already processed
    if uploaded_file.name not in st.session_state.uploaded_files:
        # Save the file to the local directory
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load document based on file type
        if uploaded_file.type == "text/plain":
            loader = TextLoader(file_path)
        elif uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(file_path)
        else:  # DOCX
            loader = Docx2txtLoader(file_path)

        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        if st.session_state.vector_store is None:
            st.session_state.vector_store = FAISS.from_documents(texts, embeddings)
        else:
            st.session_state.vector_store.add_documents(texts)

        st.session_state.vector_store.save_local("./faiss_index")

        st.success(f"Document '{uploaded_file.name}' uploaded and processed successfully!")

        # Update the list of uploaded files
        st.session_state.uploaded_files = list_uploaded_files()

        uploaded_file = None

        st.experimental_rerun()

def display_source_documents(response):
    with st.expander("Source Documents", expanded=False):
        for doc in response["context"]:
            st.markdown(f"**Source:** {os.path.basename(doc.metadata['source'])}")
            st.markdown(f"**Excerpt:** {doc.page_content}")
            st.markdown("---")
            break  # Display only the first document

user_defined_system_prompt = ""

with st.sidebar:
    # Input field for system prompt
    # user_defined_system_prompt = st.text_area("Set System Prompt:", value="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Do not answer the question from external sources apart from the documents. If no matching document is found, say you don't know. Use three sentences maximum and keep the answer concise.")
    user_defined_system_prompt = st.text_area("System Prompt:", value="You are an assistant for question-answering tasks. Use ONLY the following pieces of retrieved context to answer the question. If the answer cannot be found in the provided context, say that you don't have enough information to answer. Do not use any external knowledge. Use three sentences maximum and keep the answer concise.")
user_defined_system_prompt += "\n\n{context}"












### Contextualize question ###
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
history_aware_retriever = None

### Answer question ###
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", user_defined_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = None
qa_chain = None
rag_chain = None

if st.session_state.vector_store:
    retriever = st.session_state.vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7, "k": 5})

    # retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # retriever = custom_retriever
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    qa_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "source_documents" in message:
            display_source_documents(message["source_documents"])

history = get_session_history(session_id)

not_found_response = "Please upload some documents first to get started."
not_document_response = "No relevant document is found."

# Chat input
if prompt := st.chat_input("Your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if qa_chain:
            response = qa_chain.invoke({"input": prompt}, config={"configurable": {"session_id": session_id}})
            if not response["context"]:
                content = not_document_response
            else:
                content = response["answer"]
                
            st.markdown(content)
            message = {"role": "assistant", "content": content}
            
            if response["context"]:
                message["source_documents"] = response
                display_source_documents(response)
            
            st.session_state.messages.append(message)
        else:
            content = not_found_response
            st.markdown(content)
            st.session_state.messages.append({"role": "assistant", "content": content})