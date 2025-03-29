import os
import sys
import streamlit as st

# Add parent directory to path to allow importing when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from parent modules
from webapp import read_config, is_txt
from utils import setup_llm, compute_similarity, construct_link, check_create_symlink, hide_webapp_sidebar_item

def main():
    """
    Page for talking to your documents (RAG)
    """
    # Hide webapp sidebar item
    hide_webapp_sidebar_item()
    
    cfg = read_config()[0]
    
    # Get configuration
    RAG_TITLE = cfg.get("ui", {}).get("rag_title", "Talk to Your Documents")
    APPEND_TO_PROMPT = cfg.get("prompt", {}).get("append_to_prompt", "")
    RAG_TEXT = None
    RAG_TEXT_PATH = cfg.get("ui", {}).get("rag_text_path", None)
    PROMPT_TEMPLATE = cfg.get("prompt", {}).get("prompt_template", None)
    
    if RAG_TEXT_PATH and os.path.isfile(RAG_TEXT_PATH) and is_txt(RAG_TEXT_PATH):
        with open(RAG_TEXT_PATH, "r") as f:
            RAG_TEXT = f.read()
    
    RAG_SOURCE_PATH = cfg.get("ui", {}).get("rag_source_path", None)
    RAG_BASE_URL = cfg.get("ui", {}).get("rag_base_url", None)
    RAG_SOURCE_PATH, RAG_BASE_URL = check_create_symlink(RAG_SOURCE_PATH, RAG_BASE_URL)
    
    # Display sidebar note
    st.sidebar.markdown(
        "**Note:** Be sure to check any displayed sources to guard against hallucinations in answers."
    )
    
    # Main content
    st.header(RAG_TITLE or "Talk to Your Documents")
    
    if RAG_TEXT:
        st.markdown(RAG_TEXT, unsafe_allow_html=True)
    
    question = st.text_input(
        "Enter a question and press the `Ask` button:",
        value="",
        help="Tip: If you don't like the answer quality after pressing 'Ask', try pressing the Ask button a second time. "
        "You can also try re-phrasing the question.",
    )
    ask_button = st.button("Ask")
    llm = setup_llm()

    if question and ask_button:
        question = question + " " + APPEND_TO_PROMPT
        result = llm.ask(question, prompt_template=PROMPT_TEMPLATE)
        answer = result["answer"]
        docs = result["source_documents"]
        unique_sources = set()
        
        for doc in docs:
            answer_score = compute_similarity(answer, doc.page_content)
            question_score = compute_similarity(question, doc.page_content)
            if answer_score < 0.5 or question_score < 0.2:
                continue
            unique_sources.add(
                (
                    doc.metadata["source"],
                    doc.metadata.get("page", None),
                    doc.page_content,
                    question_score,
                    answer_score,
                )
            )
        
        unique_sources = list(unique_sources)
        unique_sources.sort(key=lambda tup: tup[-1], reverse=True)
        
        if unique_sources:
            st.markdown(
                "**One or More of These Sources Were Used to Generate the Answer:**"
            )
            st.markdown(
                "*You can inspect these sources for more information and to also guard against hallucinations in the answer.*"
            )
            
            for source in unique_sources:
                fname = source[0]
                fname = construct_link(
                    fname, source_path=RAG_SOURCE_PATH, base_url=RAG_BASE_URL
                )
                page = source[1] + 1 if isinstance(source[1], int) else source[1]
                content = source[2]
                question_score = source[3]
                answer_score = source[4]
                st.markdown(
                    f"- {fname} {', page '+str(page) if page else ''} : score: {answer_score:.3f}",
                    help=f"{content}... (QUESTION_TO_SOURCE_SIMILARITY: {question_score:.3f})",
                    unsafe_allow_html=True,
                )
        elif "I don't know" not in answer:
            st.warning(
                "No sources met the criteria to be displayed. This suggests the model may not be generating answers directly from your documents "
                + "and increases the likelihood of false information in the answer. "
                + "You should be more cautious when using this answer."
            )


if __name__ == "__main__":
    main()