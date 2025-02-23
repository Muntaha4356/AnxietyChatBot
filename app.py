import os
import gradio as gr
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

data_dir = "/home/user/app"
os.makedirs(data_dir, exist_ok=True)

pdf_filenames = [
    "mental_health_document.pdf",
    "effects_of_mental_health.pdf",
    "who_mental_health.pdf"
]

def initialize_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
    )

def create_vector_db():
    documents = []
    for filename in pdf_filenames:
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            url = f"https://huggingface.co/spaces/ANXI_BOT/{filename}"
            response = requests.get(url)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"Downloaded {filename}")
            else:
                print(f"Failed to download {filename}")
                continue
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()
    print("Chroma DB created and data saved")
    return vector_db

def load_vector_db():
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        return create_vector_db()
    else:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        return Chroma(persist_directory=db_path, embedding_function=embeddings)

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """
    You are a compassionate and professional mental health chatbot. Respond thoughtfully:
    {context}
    User: {question}
    Chatbot:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

def chatbot_response(query, chat_history=[]):
    llm = initialize_llm()
    vector_db = load_vector_db()
    qa_chain = setup_qa_chain(vector_db, llm)
    result = qa_chain.invoke({"query": query})
    chat_history.append((query, result["result"]))
    return chat_history

def gradio_interface():
    with gr.Blocks(css="""
    body {
        background-color: #F8E8EE; /* Pastel pink background */
    }
    .gradio-container {
        background: #FFF5F7; /* Light pinkish-white for contrast */
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.1);
    }
    button {
        background-color: #FFB6C1 !important; /* Pastel pink buttons */
        color: white !important;
        border-radius: 10px;
        padding: 10px;
        border: none;
    }
    """) as demo:
        gr.Markdown("""
        #  Mental Health Chatbot 
        **A gentle and supportive space for mental well-being.**
        """)
        chatbot = gr.Chatbot()
        query_input = gr.Textbox(label="Your Question:", placeholder="Type your question here...")
        submit_btn = gr.Button("Ask")
        submit_btn.click(fn=chatbot_response, inputs=[query_input, chatbot], outputs=chatbot)
    demo.launch(share=True)

gradio_interface()
