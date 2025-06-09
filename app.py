import streamlit as st
from dotenv import load_dotenv
from utils import *
import uuid
import os

if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] = ''

def main():
    load_dotenv()
    st.set_page_config(page_title="Resume Screening Assistance")
    st.title("HR - Resume Screening Assistance...💁")
    st.subheader("I can help you in resume screening process")

    job_description = st.text_area("Please paste the 'JOB DESCRIPTION' here...", key="1")
    document_count = st.text_input("No. of 'RESUMES' to return", key="2")
    pdf = st.file_uploader("Upload resumes here, only PDF files allowed", type=["pdf"], accept_multiple_files=True)

    submit = st.button("Help me with the analysis")

    if submit:
        if not pdf or not job_description or not document_count:
            st.error("Please fill all the required fields.")
            return

        with st.spinner('Processing...'):
            st.session_state['unique_id'] = uuid.uuid4().hex
            final_docs_list = create_docs(pdf, st.session_state['unique_id'])
            st.write(f"*Resumes uploaded*: {len(final_docs_list)}")

            # Debug print metadata of docs
            for d in final_docs_list:
                st.write(f"Metadata: {d.metadata}")

            embeddings = create_embeddings_load_data()

            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
            pinecone_index = os.getenv("PINECONE_INDEX_NAME")

            if not all([pinecone_api_key, pinecone_env, pinecone_index]):
                st.error("Missing Pinecone credentials in environment variables!")
                return

            push_to_pinecone(pinecone_api_key, pinecone_env, pinecone_index, embeddings, final_docs_list)

            try:
                # Remove unique_id filter temporarily for debugging
                relevant_docs = similar_docs(
                    job_description, document_count,
                    pinecone_api_key, pinecone_env,
                    pinecone_index, embeddings,
                    st.session_state['unique_id'],
                    use_filter=False  # added param to toggle filter
                )
                st.write(f"Retrieved {len(relevant_docs)} similar resumes.")
            except Exception as e:
                st.error(f"Error fetching similar documents: {e}")
                return

            if not relevant_docs:
                st.warning("No relevant resumes matched the job description.")
                return

            st.write("─" * 30)

            for idx, (doc, score) in enumerate(relevant_docs, 1):
                st.subheader(f"👉 {idx}")
                st.write(f"**File** : {doc.metadata.get('name', 'Unknown')}")

                with st.expander("Show me 👀"):
                    st.info(f"**Match Score** : {score:.4f}")
                    try:
                        summary = get_summary(doc)
                        st.write(f"**Summary** : {summary}")
                    except Exception as e:
                        st.warning(f"Summary generation failed: {e}")

        st.success("Hope I was able to save your time")

if __name__ == '__main__':
    main()
