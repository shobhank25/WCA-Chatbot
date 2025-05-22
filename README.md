# 🧊 WCA Regulations Chatbot

A conversational AI application built with LangChain, ChromaDB, and Streamlit to help users interactively query the official rules of the World Cube Association (WCA). This chatbot intelligently answers questions using up-to-date WCA regulation documents and provides contextual references for transparency.
🚀 Features

    🔎 Retrieval-Augmented Generation (RAG) using WCA’s official documents

    💬 Streamlit-powered chat interface

    🧠 Uses sentence-transformer embeddings for semantic understanding

    🗃️ Fast and persistent vector search via ChromaDB

    🌐 Sources live data from:

        WCA Regulations

        WCA Guidelines

        WCA Scrambles

🧰 Tech Stack

    LangChain

    Streamlit

    ChromaDB

    sentence-transformers

    Unstructured

    OpenAI API (via OpenRouter)

📦 Installation

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
streamlit run app.py

    🗝️ Make sure to set your OpenRouter API key in the OPENROUTER_API_KEY variable inside app.py.

🧠 How It Works

    Loads WCA regulation pages using UnstructuredURLLoader

    Cleans and splits text using RecursiveCharacterTextSplitter

    Embeds the content using sentence-transformers

    Stores embeddings in a Chroma vector database

    Responds to user queries using LangChain’s RetrievalQA pipeline
