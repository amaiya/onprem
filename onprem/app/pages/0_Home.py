import os
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to allow importing when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from parent modules
from webapp import read_config
from utils import hide_webapp_sidebar_item

def main():
    """
    Home page for the application
    """
    # Hide webapp sidebar item
    hide_webapp_sidebar_item()
    
    cfg = read_config()[0]
    
    st.header("Welcome to OnPrem.LLM")
    
    # Load LLM to get model name
    from onprem.app.utils import load_llm
    llm = load_llm()
    model_name = llm.model_name
    
    # Main content
    st.markdown("""
    This application allows you to interact with an LLM in multiple ways:
    
    1. **Use Prompts to Solve Problems**: Submit prompts directly to the LLM model
    2. **Talk to Your Documents**: Ask questions about your documents using RAG technology
    3. **Search Documents**: Search through your indexed documents using keywords or semantic search
    4. **Manage**: Upload documents, manage folders, and configure the application settings
    
    Use the sidebar to navigate between these different features.
    """)
    
    # Model information
    st.subheader("Current Model Information")
    st.markdown(f"**Model**: {model_name}")
    
    # Quick links
    st.subheader("Quick Links")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💬 Use Prompts", use_container_width=True):
            st.switch_page("pages/1_Prompts.py")
    
    with col2:
        if st.button("📄 Talk to Documents", use_container_width=True):
            st.switch_page("pages/2_Documents.py")
    
    with col3:
        if st.button("🔍 Search Documents", use_container_width=True):
            st.switch_page("pages/4_Search.py")
    
    # Additional information
    st.markdown("---")
    st.markdown("Built with [OnPrem](https://github.com/amaiya/onprem)")


if __name__ == "__main__":
    # Set page to wide mode when run directly
    st.set_page_config(
        page_title="Home", 
        page_icon="🏠", 
        layout="centered",
        initial_sidebar_state="expanded"
    )
    main()
