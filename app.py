import os
import shutil

import gradio as gr

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


# Gemini setup

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# Global variables

database = None
chat_history = []


# Create embeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)



def process_pdf(pdf_file):

    global database

    if os.path.exists("vectorstore"):
        shutil.rmtree("vectorstore")


    loader = PyPDFLoader(pdf_file)

    documents = loader.load()


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )


    chunks = splitter.split_documents(documents)


    database = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="vectorstore"
    )


    return "PDF processed successfully! You can ask questions now."



def ask_question(question):

    global chat_history


    if database is None:
        return "Please upload and process a PDF first."


    results = database.similarity_search(
        question,
        k=3
    )


    context = ""

    for result in results:
        context += result.page_content + "\n"


    history_text = ""

    for chat in chat_history:
        history_text += (
            "User: " + chat["question"] +
            "\nAssistant: " + chat["answer"] +
            "\n"
        )


    prompt = f"""
You are a helpful PDF assistant.

Use only the given context to answer.

Previous conversation:
{history_text}


Context:
{context}


Question:
{question}
"""


    response = llm.invoke(prompt)


    answer = response.content


    chat_history.append(
        {
            "question": question,
            "answer": answer
        }
    )


    return answer



# Gradio Interface


with gr.Blocks() as app:

    gr.Markdown(
        "# PDF Question Answering Assistant"
    )


    pdf = gr.File(
        label="Upload PDF",
        type="filepath"
    )


    process_button = gr.Button(
        "Process PDF"
    )


    status = gr.Textbox(
        label="Status"
    )


    process_button.click(
        process_pdf,
        inputs=pdf,
        outputs=status
    )


    question = gr.Textbox(
        label="Ask Question"
    )


    answer = gr.Textbox(
        label="Answer"
    )


    ask_button = gr.Button(
        "Ask"
    )


    ask_button.click(
        ask_question,
        inputs=question,
        outputs=answer
    )


app.launch()