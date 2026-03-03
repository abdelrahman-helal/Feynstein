#!/usr/bin/env python3
"""
Script to process physics textbooks once and save the vector database.
This should be run once before deploying to Vercel or other platforms.
"""

import os
from utils.langchain_rag_system import langchain_rag

def main():
    print("=== Feynstein Textbook Processing ===")
    print("This script will process all PDF textbooks in the 'textbooks' directory")
    print("and create a vector database for the RAG system.")
    print()
    
    # Check if textbooks directory exists
    textbooks_dir = "textbooks"
    if not os.path.exists(textbooks_dir):
        print(f"Error: '{textbooks_dir}' directory not found!")
        print("Please create the directory and add your PDF textbooks.")
        return
    
    # List available textbooks
    pdf_files = [f for f in os.listdir(textbooks_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{textbooks_dir}' directory!")
        print("Please add your physics textbook PDFs to the directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf}")
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed with processing these textbooks? (y/N): ")
    if response.lower() != 'y':
        print("Processing cancelled.")
        return
    
    print("\nStarting textbook processing...")
    print("This may take several minutes depending on the size of your textbooks.")
    print()
    
    try:
        # Process and save vector database
        langchain_rag.process_and_save_vector_db(textbooks_dir)
        
        print("\n=== Processing Complete! ===")
        print("You can now deploy your application to Vercel or other platforms.")
        print("The vector database will be loaded automatically when the app starts.")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        print("Please check your PDF files and try again.")

if __name__ == "__main__":
    main() 