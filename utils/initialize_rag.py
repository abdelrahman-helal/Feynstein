import os
from pathlib import Path

def initialize_rag_system(rag_system):
    """
    Initialize the RAG system by processing all physics textbooks.
    """
    # Create textbooks directory if it doesn't exist
    textbooks_dir = Path("textbooks")
    
    # Dictionary mapping textbook IDs to their expected filenames
    textbook_files = {
        "openstax_vol1": "UniversityPhysicsVol1.pdf",
        "openstax_vol2": "UniversityPhysicsVol2.pdf",
        "openstax_vol3": "UniversityPhysicsVol3.pdf",
        "griffiths_qm": "IntroductionToQM.pdf",
        "griffiths_em": "IntroductionToElectrodynamics.pdf",
        "taylor_cm": "ClassicalMechanics.pdf"
    }
    
    print("Initializing RAG system with physics textbooks...")
    
    for textbook_id, filename in textbook_files.items():
        file_path = textbooks_dir / filename
        if file_path.exists():
            print(f"Processing {filename}...")
            try:
                rag_system.process_textbook(str(file_path), textbook_id)
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        else:
            print(f"Warning: {filename} not found in textbooks directory")
    
    print("\nRAG system initialization complete!")
    print("\nPlease ensure you have the following PDF files in the 'textbooks' directory:")
    for filename in textbook_files.values():
        print(f"- {filename}")

if __name__ == "__main__":
    initialize_rag_system() 