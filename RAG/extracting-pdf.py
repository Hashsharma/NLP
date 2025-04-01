import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the Sentence Transformer model and load it onto the device (GPU or CPU)
model = SentenceTransformer("D:/mr_document/all_models/all-MiniLM-L6-v2-original/")  # You can choose a different model if needed
model.to(device)  # Move model to GPU if available, else CPU

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Extract text from each page
    pdf_text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pdf_text += page.get_text()
    
    return pdf_text

# Function to get sentence embeddings using Sentence Transformer
def get_embeddings(text):
    # Split the text into sentences
    sentences = text.split(". ")
    
    # Get embeddings for each sentence
    embeddings = model.encode(sentences, device=device)  # Pass device to ensure the computation happens on the correct device
    
    return sentences, embeddings

# Main function
def process_pdf_and_get_embeddings(pdf_path):
    # Step 1: Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Get embeddings from the extracted text
    sentences, embeddings = get_embeddings(pdf_text)
    
    # Example: store or print the embeddings and sentences (Here we print)
    for sentence, embedding in zip(sentences, embeddings):
        print(f"Sentence: {sentence}")
        print(f"Embedding: {embedding[:10]}...")  # Print first 10 elements of the embedding for brevity

    # You can save embeddings to a file or a database if needed.
    # For example, you can save them as numpy arrays or in a CSV file.
    
    # Optionally save embeddings to a file (e.g., .npy for numpy arrays)
    # np.save('embeddings.npy', embeddings)  # Save embeddings to a file for later use.

    return sentences, embeddings


# Example usage
pdf_path = '../resource/pdf_to_extract.pdf'  # Replace with the actual path to your PDF file
sentences, embeddings = process_pdf_and_get_embeddings(pdf_path)
