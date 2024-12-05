import sys
import os
import streamlit as st
from pathlib import Path
import shutil  # For directory management

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.embedding import download_gemini_embedding
from QAWithPDF.model_api import load_model


def main():
    st.set_page_config(page_title="QA with Documents")

    # File uploader
    doc = st.file_uploader("Upload your document", type=["pdf", "txt", "docx"])

    st.header("QA with Documents (Information Retrieval)")

    user_question = st.text_input("Ask your question")

    if st.button("Submit & Process"):
        if doc is not None:
            with st.spinner("Processing..."):
                # Create a temporary directory
                temp_dir = Path("temp_directory")
                temp_dir.mkdir(exist_ok=True)

                # Save the uploaded file into the temporary directory
                temp_file_path = temp_dir / doc.name
                with open(temp_file_path, "wb") as f:
                    f.write(doc.read())

                try:
                    # Pass the directory to `load_data`
                    document = load_data(str(temp_dir))

                    # Load the model
                    model = load_model()

                    # Generate the query engine
                    query_engine = download_gemini_embedding(model, document)

                    # Query the engine
                    response = query_engine.query(user_question)

                    # Display the response
                    st.write(response.response)

                except Exception as e:
                    st.error(f"An error occurred: {e}")

                finally:
                    # Clean up the temporary directory and file
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)

        else:
            st.warning("Please upload a document before submitting.")


if __name__ == "__main__":
    main()
