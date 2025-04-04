import os
import sys
import streamlit as st
from typing import Optional, Tuple, List, Dict, Any
import mimetypes
import uuid

# Add parent directory to path to allow importing when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from parent modules
from webapp import read_config, is_txt
from utils import setup_llm, load_llm, compute_similarity, construct_link, check_create_symlink, hide_webapp_sidebar_item

def get_file_data(filepath: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Read a file and determine its MIME type
    
    Args:
        filepath: Path to the file to read
        
    Returns:
        Tuple containing:
        - The file data as bytes (or None if the file couldn't be read)
        - The MIME type (or a default type if it couldn't be determined)
    """
    try:
        with open(filepath, 'rb') as f:
            file_data = f.read()
            
        mime_type, _ = mimetypes.guess_type(filepath)
        if mime_type is None:
            # Default to plain text if MIME type can't be determined
            mime_type = 'application/octet-stream'
            
        return file_data, mime_type
    except Exception as e:
        st.error(f"Error reading file {filepath}: {str(e)}")
        return None, None


def create_document_display(
    filepath: str, 
    source_path: Optional[str], 
    base_url: Optional[str], 
    page: Optional[int] = None,
    score: Optional[float] = None
) -> str:
    """
    Creates either a hyperlink (on Unix) or a download button (on Windows) for a document
    
    Args:
        filepath: Path to the document
        source_path: Base path for resolving relative paths
        base_url: Base URL for hyperlinks
        page: Page number (for PDFs)
        score: Similarity score to display
        
    Returns:
        HTML string for the document link/button
    """
    filename = os.path.basename(filepath)
    page_info = f", page {page + 1}" if isinstance(page, int) else ""
    score_info = f" : score: {score:.3f}" if score is not None else ""
    
    if os.name == 'nt':  # Windows
        # For Windows, use Streamlit's download button with a unique key
        button_id = f"download_{uuid.uuid4().hex}"
        st.session_state[button_id] = {
            'filepath': filepath,
            'filename': filename
        }
        
        # Create a HTML div for the Windows version
        return f"""<div id="{button_id}_wrapper">
                  <span><strong>{filename}</strong>{page_info}{score_info}</span>
               </div>"""
    else:
        # For Unix, use the standard hyperlink approach
        return construct_link(filepath, source_path, base_url) + f"{page_info}{score_info}"


def main():
    """
    Page for talking to your documents (RAG)
    """
    # Hide webapp sidebar item
    hide_webapp_sidebar_item()
    
    # Initialize session state for persisting results across page reloads
    if 'question' not in st.session_state:
        st.session_state.question = ""
    if 'answer' not in st.session_state:
        st.session_state.answer = None
    if 'unique_sources' not in st.session_state:
        st.session_state.unique_sources = []
    
    cfg = read_config()[0]
    
    # Get configuration
    RAG_TITLE = cfg.get("ui", {}).get("rag_title", "Talk to Your Documents")
    APPEND_TO_PROMPT = cfg.get("prompt", {}).get("append_to_prompt", "")
    RAG_TEXT = None
    RAG_TEXT_PATH = cfg.get("ui", {}).get("rag_text_path", None)
    PROMPT_TEMPLATE = cfg.get("prompt", {}).get("prompt_template", None)
    
    # Load LLM to get model name
    llm = setup_llm()
    MODEL_NAME = llm.model_name
    
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
    
    # Add some CSS for better styling - same as in Prompts.py
    st.markdown("""
    <style>
    /* Improve chat message styling */
    .stChatMessage {
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Style the chat input */
    .stChatInputContainer {
        border-top: 1px solid #e0e0e0;
        padding-top: 1rem;
    }
    
    /* Make user messages stand out */
    .stChatMessage[data-testid="stChatMessage-user"] {
        background-color: #f0f7ff;
    }
    
    /* Style the assistant messages */
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background-color: #f9f9f9;
    }
    
    /* System messages should be subtle */
    .stChatMessage[data-testid="stChatMessage-system"] {
        background-color: #f0f0f0;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

    # Improved page header - same style as in Prompts.py
    st.markdown(f"""
    <h1 style="
        color: #0068c9;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0068c9;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
    ">
        <span style="font-size: 1.8rem; margin-right: 0.5rem;">📚</span> 
        {RAG_TITLE or "Talk to Your Documents"}
    </h1>
    """, unsafe_allow_html=True)
    
    # Enhanced model info card - same as in Prompts.py
    st.markdown(f"""
    <div style="
        padding: 12px 15px;
        border-radius: 8px;
        margin-bottom: 25px;
        background: linear-gradient(to right, rgba(0, 104, 201, 0.05), rgba(92, 137, 229, 0.1));
        border-left: 4px solid #0068c9;
        display: flex;
        align-items: center;
    ">
        <span style="font-size: 1.2rem; margin-right: 10px;">🤖</span>
        <div>
            <p style="margin: 0; font-weight: 500;">Current Model: <span style="color: #0068c9;">{MODEL_NAME}</span></p>
            <p style="margin: 3px 0 0 0; font-size: 0.85rem; color: #666;">
                Ask questions about your documents to get AI-generated answers based on their content.
            </p>
        </div>
    </div>
    """  , unsafe_allow_html=True)
    
    if RAG_TEXT:
        st.markdown(RAG_TEXT, unsafe_allow_html=True)
    
    question = st.text_input(
        "Enter a question and press the `Ask` button:",
        value=st.session_state.question,
        help="Tip: If you don't like the answer quality after pressing 'Ask', try pressing the Ask button a second time. "
        "You can also try re-phrasing the question.",
        key="question_input"
    )
    
    # Create a container for the buttons
    col1, col2 = st.columns([5, 1])
    
    # Reset button to clear results
    def reset_results():
        st.session_state.question = ""
        st.session_state.answer = None
        st.session_state.unique_sources = []
    
    with col1:
        ask_button = st.button("Ask", type='primary')
    
    with col2:
        reset_button = st.button("Reset", type='secondary', on_click=reset_results)
    
    llm = setup_llm()

    # Process new query when Ask button is clicked
    first_answer = False if st.session_state.answer else True
    if question and ask_button:
        # Save the question to session state
        st.session_state.question = question
        
        # Create a placeholder for the streaming output
        stream_placeholder = st.empty()
        
        with stream_placeholder:
            st.info("Generating response...")
        
        # Process the question with temporarily hidden output
        with stream_placeholder:
            # Get a new LLM instance from the loaded model
            muted_llm = load_llm()
            # Temporarily disable streaming by replacing callbacks
            original_callbacks = muted_llm.callbacks
            muted_llm.callbacks = []
            
            # Process the question
            question_with_append = question + " " + APPEND_TO_PROMPT
            result = muted_llm.ask(question_with_append, prompt_template=PROMPT_TEMPLATE)
            
            # Restore original callbacks
            muted_llm.callbacks = original_callbacks
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
        
        # Save results to session state
        st.session_state.answer = answer
        st.session_state.unique_sources = unique_sources
        
        # Clear the streaming placeholder
        stream_placeholder.empty()
    
    # Display answer if it exists (either from new query or from session state)
    if st.session_state.answer and not first_answer:
        st.markdown(f"**Answer:**\n{st.session_state.answer}")
        
    # Display sources if they exist (either from new query or from session state)
    if st.session_state.unique_sources:
        st.markdown(
            "**One or More of These Sources Were Used to Generate the Answer:**"
        )
        st.markdown(
            "*You can inspect these sources for more information and to also guard against hallucinations in the answer.*"
        )
        
        # On Windows, we'll need to add download buttons separately
        is_windows = os.name == 'nt'
        
        for i, source in enumerate(st.session_state.unique_sources):
            filepath = source[0]
            page = source[1]
            content = source[2]
            question_score = source[3]
            answer_score = source[4]
            
            # Create either a hyperlink (Unix) or a placeholder + download button (Windows)
            doc_display = create_document_display(
                filepath=filepath,
                source_path=RAG_SOURCE_PATH,
                base_url=RAG_BASE_URL,
                page=page,
                score=answer_score
            )
            
            # Add the markdown content
            st.markdown(
                f"- {doc_display}",
                help=f"{content}... (QUESTION_TO_SOURCE_SIMILARITY: {question_score:.3f})",
                unsafe_allow_html=True,
            )
            
            # For Windows, add download buttons
            if is_windows:
                # Get unique ID for this source
                button_id = f"source_btn_{i}"
                
                # Read the file data
                file_data, mime_type = get_file_data(filepath)
                
                if file_data:
                    # Add download button for this source
                    st.download_button(
                        label=f"📄 Download {os.path.basename(filepath)}",
                        data=file_data,
                        file_name=os.path.basename(filepath),
                        mime=mime_type,
                        key=button_id
                    )
    elif st.session_state.answer and "I don't know" not in st.session_state.answer:
        st.warning(
            "No sources met the criteria to be displayed. This suggests the model may not be generating answers directly from your documents "
            + "and increases the likelihood of false information in the answer. "
            + "You should be more cautious when using this answer."
        )


if __name__ == "__main__":
    # Set page to wide mode when run directly
    st.set_page_config(
        page_title="Documents", 
        page_icon="📚", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    main()
