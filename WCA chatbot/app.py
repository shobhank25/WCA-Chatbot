import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import re
import os

# Configuration
WCA_REGULATION_URLS = [
    "https://www.worldcubeassociation.org/regulations/",
    "https://www.worldcubeassociation.org/regulations/guidelines.html",
    "https://www.worldcubeassociation.org/regulations/scrambles/"
]

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
PERSIST_DIRECTORY = "db"
OPENROUTER_API_KEY = "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"

PROMPT_TEMPLATE = """
You are a WCA regulations expert. Use the context to answer questions.

Context: {context}

Question: {question}

Answer:
"""

def load_and_preprocess_documents():
    """Load and preprocess WCA regulation documents"""
    loader = UnstructuredURLLoader(urls=WCA_REGULATION_URLS)
    data = loader.load()
    
    # Clean content
    for doc in data:
        doc.page_content = re.sub(r"\n{3,}", "\n", doc.page_content)
        doc.page_content = re.sub(r" {2,}", " ", doc.page_content)
    
    return data

def setup_qa_system():
    """Set up the entire QA system"""
    with st.spinner("Loading WCA regulations..."):
        try:
            # Load and process documents
            documents = load_and_preprocess_documents()
            
            # Set up embeddings and vector store
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            vectordb = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
            vectordb.persist()
            
            # Set up QA chain
            os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
            os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
            
            llm = ChatOpenAI(
                model_name="deepseek/deepseek-chat",
                temperature=0.7,
                max_tokens=2048,
                top_p=0.95
            )
            
            prompt = PromptTemplate(
                template=PROMPT_TEMPLATE,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            return qa_chain
        except Exception as e:
            st.error(f"Failed to initialize: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="WCA Regulations Chatbot",
        page_icon="🧊",
        layout="centered"
    )
    
    # Initialize session state
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = setup_qa_system()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        if st.session_state.qa_chain:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I'm ready to answer questions about WCA regulations!"
            })
    
    # Display chat interface
    st.title("🧊 WCA Regulations Chatbot")
    st.caption("Ask about official competition rules from the World Cube Association")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Regulation References"):
                    for i, source in enumerate(message["sources"], 1):
                        st.caption(f"Source {i}: {source[:200]}...")
    
    # Handle user input
    if prompt := st.chat_input("Ask about WCA regulations..."):
        if not st.session_state.qa_chain:
            st.warning("System still initializing...")
            return
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.spinner("Checking regulations..."):
            try:
                result = st.session_state.qa_chain({"query": prompt})
                
                # Add assistant response to chat history
                response_content = result['result']
                sources = [doc.page_content for doc in result["source_documents"]]
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_content,
                    "sources": sources
                })
                
                # Display response
                with st.chat_message("assistant"):
                    st.markdown(response_content)
                    with st.expander("Regulation References"):
                        for i, source in enumerate(sources, 1):
                            st.caption(f"• Source {i}: {source[:200]}...")
                            
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")

if __name__ == "__main__":
    main()