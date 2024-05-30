import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from Template import css, bot_template, user_template
import base64
from googlesearch import search
import concurrent.futures
from collections import Counter
import nltk
from nltk.corpus import stopwords

def MergePdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def SplitPdf(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def FaissVectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='bert-base-uncased')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def ResponseChain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def update_model(response, user_question, feedback):
    print(f"User question: {user_question}")
    print(f"Bot response: {response}")
    print(f"Feedback: {feedback}")

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    i = 0
    while i < len(st.session_state.chat_history):
        message = st.session_state.chat_history[i]
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            feedback = st.selectbox(f"Was this response helpful? {i}", ["Yes", "No"])
            if feedback == "No":
                query = f"{user_question} site:wikipedia.org"
                try:
                    google_result = next(search(query, num=1, stop=1))
                    st.write(f"According to Google, {google_result}")
                except StopIteration:
                    st.write("Sorry, I couldn't find any information on Google.")
            elif feedback == "Yes":
                i += 2
        i += 1

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_suggestion_prompts(text_chunks):
    suggestions = []
    all_text = ' '.join(text_chunks)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in nltk.word_tokenize(all_text.lower()) if word not in stop_words]
    
    # Get word counts and most common words
    word_counts = Counter(words)
    common_words = [word for word, count in word_counts.most_common(20) if len(word) > 3]  # Exclude short words

    # Generate suggestions based on common words
    for word in common_words[:3]:
        suggestions.append(f"What information is available about {word}?")
    
    return suggestions

def get_document_summaries(text_chunks):
    summaries = []
    for chunk in text_chunks[:5]:
        response = st.session_state.conversation({'question': f"Summarize the following text: {chunk}"})
        summary = response['chat_history'][-1].content
        summaries.append(summary)
    return summaries

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    img = get_img_as_base64("img1.jpg")
    img1 = get_img_as_base64("bg1.jpg")

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{img1}");
    background-color: rgba(255, 255, 255, 0.5);
    background-size: cover;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
    }}

    [data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{img}");
    backgroud-size: cover
    background-position: center; 
    background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}

    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    if "suggestion_prompts" not in st.session_state:
        st.session_state.suggestion_prompts = []

    text_chunks = []  # Initialize text_chunks as an empty list

    st.header("PDF Insighter\n ChatBot for Document Intelligence:books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        st.session_state.query_history.append(user_question)
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    merged_texts = executor.map(MergePdf, [[pdf] for pdf in pdf_docs])
                    Merged_pdf_text = ''.join(merged_texts)

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    text_chunks_lists = executor.map(SplitPdf, [Merged_pdf_text])
                    text_chunks = [chunk for chunks in text_chunks_lists for chunk in chunks]

                vectorstore = FaissVectorstore(text_chunks)
                st.session_state.conversation = ResponseChain(vectorstore)

                # Generate suggestion prompts
                if not st.session_state.suggestion_prompts:
                    st.session_state.suggestion_prompts = get_suggestion_prompts(text_chunks)

    # Display suggestion prompts
    st.sidebar.subheader("Suggestion Prompts")
    for prompt in st.session_state.suggestion_prompts:
        st.sidebar.write(prompt)

if __name__ == '__main__':
    nltk.download('stopwords')
    main()
