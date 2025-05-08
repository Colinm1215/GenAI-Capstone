import docx
import fitz
import requests
import streamlit as st


API_URL = "https://Jack1224-CapstoneBackendV2.hf.space/analyze"

st.set_page_config(
    page_title="LLM-Powered News Analyzer",
    layout="wide"
)

st.markdown("""
    <div style='text-align: center; padding-bottom: 1rem;'>
        <h1>News Bias Detection</h1>
        <p>LLM tool to summarize and detect biases within news articles</p>
    </div>
""", unsafe_allow_html=True)

left_col, center_col, right_col = st.columns([1, 2, 1])
with center_col:
    uploaded_files = st.file_uploader(
        "Upload a news article (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

if uploaded_files:
    for file in uploaded_files:
        file_name = file.name
        file_bytes = file.read()
        st.divider()
        with st.expander(f"File: {file_name}", expanded=True):
            with st.spinner("Extracting text..."):
                file_type = file_name.split('.')[-1]
                match file_type:
                    case 'pdf':
                        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                            file_text = "\n".join([page.get_text() for page in doc])
                    case 'docx':
                        doc = docx.Document(file)
                        file_text = "\n".join([para.text for para in doc.paragraphs])
                    case 'txt':
                        file_text = file_bytes.decode("utf-8")
                    case _:
                        st.error("Unsupported file type")
                assert isinstance(file_text, str) and len(file_text.strip()) > 0

            with st.container():

                st.subheader("Original Article :")
                st.write(file_text)
            st.divider()
            with st.spinner("Analyzing and Summarizing Article..."):
                try:
                    response = requests.post(API_URL, json={"text": file_text})
                    response.raise_for_status()
                    data = response.json()
                except Exception as e:
                    st.error(f"API request failed: {e}")
                    continue

                summary = data.get("summary", "")
                sentiment_results = data.get("bias", {})

                st.subheader("Summary Article :")
                st.write(summary)

                st.divider()

                if sentiment_results:
                    st.subheader("Sentiment per Named Entity")
                    for ent, result in sentiment_results.items():
                        col1, col2 = st.columns([2, 1])
                        col1.markdown(f"[{ent}]")
                        col2.markdown(f"Sentiment: `{result['sentiment']}`")

                else:
                    st.write("No sentiment results detected, or an error occurred.")
