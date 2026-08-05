##  PDF Question Answering Assistant using RAG

Name: Josin Joji
MUID: josinjoji@mulearn

### Project Overview

This project is a PDF Question Answering Application built using Retrieval-Augmented Generation (RAG).
The application allows users to upload a PDF document and ask questions related to its content. The system reads the PDF, converts the text into smaller chunks, creates embeddings, stores them in a vector database, and retrieves relevant information to generate accurate answers using a Large Language Model.

The application also maintains conversation history so users can ask follow-up questions naturally.

### Technologies Used

* Python - Programming language
* LangChain - Framework for building LLM applications
* PyPDFLoader - Loading and extracting text from PDF files
* Sentence Transformers - Generating text embeddings
* ChromaDB - Vector database for storing embeddings
* Google Gemini API - Free LLM for generating answers
* Gradio- Interactive user interface
* python-dotenv - Managing environment variables

### Memory Implementation

Conversation memory is implemented using a chat history system.

The application stores previous user questions and assistant responses. This allows the assistant to understand follow-up questions based on previous conversations.

Example:

User: What is this document about?

Assistant: This document explains a software license agreement.

User: Who created it?

Assistant: The document was created by Blackmagic Design.

### Challenges Faced

* Managing compatibility between different LangChain versions
* Setting up the Python virtual environment
* Handling PDF text extraction and chunking
* Connecting the Gemini API correctly
* Understanding the complete RAG workflow
* Managing vector storage using ChromaDB

## Future Improvements

* Add support for multiple PDF uploads
* Add OCR support for scanned PDFs
* Improve chat interface design
* Add source citation for answers
* Deploy the application with cloud hosting
* Add user authentication