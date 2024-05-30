import streamlit as st
print("Streamlit import successful")

try:
    from dotenv import load_dotenv
    print("dotenv import successful")
except ImportError:
    print("dotenv import failed")

try:
    from PyPDF2 import PdfReader
    print("PyPDF2 import successful")
except ImportError:
    print("PyPDF2 import failed")

try:
    from langchain.text_splitter import CharacterTextSplitter
    print("text_splitter import successful")
except ImportError:
    print("text_splitter import failed")

try:
    from langchain.embeddings import HuggingFaceEmbeddings
    print("HuggingFaceEmbeddings import successful")
except ImportError:
    print("HuggingFaceEmbeddings import failed")

try:
    from langchain.vectorstores import FAISS
    print("FAISS import successful")
except ImportError:
    print("FAISS import failed")

try:
    from langchain.chat_models import ChatOpenAI
    print("ChatOpenAI import successful")
except ImportError:
    print("ChatOpenAI import failed")

try:
    from langchain.memory import ConversationBufferMemory
    print("ConversationBufferMemory import successful")
except ImportError:
    print("ConversationBufferMemory import failed")

try:
    from langchain.chains import ConversationalRetrievalChain
    print("ConversationalRetrievalChain import successful")
except ImportError:
    print("ConversationalRetrievalChain import failed")

try:
    from Template import css, bot_template, user_template
    print("Template import successful")
except ImportError:
    print("Template import failed")

try:
    import base64
    print("base64 import successful")
except ImportError:
    print("base64 import failed")

try:
    from googlesearch import search
    print("googlesearch import successful")
except ImportError:
    print("googlesearch import failed")

try:
    import concurrent.futures
    print("concurrent.futures import successful")
except ImportError:
    print("concurrent.futures import failed")

try:
    from collections import Counter
    print("Counter import successful")
except ImportError:
    print("Counter import failed")

try:
    import nltk
    print("nltk import successful")
except ImportError:
    print("nltk import failed")

try:
    from nltk.corpus import stopwords
    print("nltk stopwords import successful")
except ImportError:
    print("nltk stopwords import failed")
