import os
import sys
import yaml
import streamlit as st
from pathlib import Path

# Add parent directory to path to allow importing when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from parent modules
from webapp import DEFAULT_YAML_FPATH, read_config, DEFAULT_YAML
from utils import hide_webapp_sidebar_item, load_llm

def main():
    """
    Settings page for configuring the application
    """
    # Hide webapp sidebar item
    hide_webapp_sidebar_item()
    
    st.header("Settings")
    
    # Create tabs for different settings
    tab1, tab2 = st.tabs(["Configuration", "Document Ingestion"])
    
    with tab1:
        st.subheader("Configuration File")
        
        cfg, _ = read_config()
        
        # Display and edit current configuration
        st.info(f"Configuration file location: {DEFAULT_YAML_FPATH}")
        
        # Edit YAML directly
        with open(DEFAULT_YAML_FPATH, "r") as f:
            yaml_content = f.read()
        
        edited_yaml = st.text_area("Edit Configuration", yaml_content, height=400)
        
        if st.button("Save Configuration"):
            try:
                # Validate YAML format by parsing it
                yaml.safe_load(edited_yaml)
                
                # Save the edited YAML
                with open(DEFAULT_YAML_FPATH, "w") as f:
                    f.write(edited_yaml)
                
                st.success("Configuration saved successfully. Please restart the app for changes to take effect.")
            except Exception as e:
                st.error(f"Error saving configuration: {str(e)}")
        
        # Reset to default
        if st.button("Reset to Default Configuration"):
            try:
                from onprem import utils as U
                yaml_content = DEFAULT_YAML.format(datadir=U.get_datadir()).strip()
                yaml_content = yaml_content.replace('PROMPT_VARIABLE', '{prompt}')
                
                with open(DEFAULT_YAML_FPATH, "w") as f:
                    f.write(yaml_content)
                
                st.success("Configuration reset to default. Please restart the app for changes to take effect.")
            except Exception as e:
                st.error(f"Error resetting configuration: {str(e)}")
    
    with tab2:
        st.subheader("Document Ingestion")
        
        # Instructions
        st.markdown("""
        This tool allows you to ingest documents from a folder into the vector database.
        Select a folder containing documents to ingest, configure parameters, and click "Ingest Documents".
        
        Supported formats include PDF, DOCX, TXT, HTML, CSV, and more.
        """)
        
        # Folder selection
        folder_path = st.text_input("Folder Path", 
                                   placeholder="Enter the absolute path to your documents folder",
                                   help="Enter the full path to the folder containing documents to ingest")
        
        # Chunking settings
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input("Chunk Size", 
                                       min_value=100, 
                                       max_value=2000, 
                                       value=500,
                                       help="Text is split into chunks of this many characters")
        with col2:
            chunk_overlap = st.number_input("Chunk Overlap", 
                                          min_value=0, 
                                          max_value=500, 
                                          value=50,
                                          help="Character overlap between chunks")
        
        # Get store type from config
        cfg = read_config()[0]
        store_type = cfg.get("llm", {}).get("store_type", "dense")
        
        # Display store type (read-only)
        st.info(f"Using store type from configuration: {store_type}")
        
        # Batch size
        batch_size = st.number_input("Batch Size", 
                                    min_value=1, 
                                    max_value=5000, 
                                    value=1000,
                                    help="Number of documents to process in each batch")
        
        # Ingest button
        if st.button("Ingest Documents"):
            if not folder_path:
                st.error("Please enter a valid folder path")
            elif not os.path.isdir(folder_path):
                st.error(f"Directory does not exist: {folder_path}")
            else:
                try:
                    with st.spinner(f"Ingesting documents from {folder_path}..."):
                        # Load the LLM with the current configuration
                        llm = load_llm()
                        
                        # The store_type is already set from the config in load_llm()
                        
                        # Ingest documents
                        result = llm.ingest(
                            source_directory=folder_path,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            batch_size=batch_size
                        )
                        
                        # Show success message
                        st.success(f"Successfully ingested documents from {folder_path}")
                        
                        # Show stats if available
                        if isinstance(result, dict) and "num_added" in result:
                            st.info(f"Added {result['num_added']} new document chunks to the vector database")
                except Exception as e:
                    st.error(f"Error during document ingestion: {str(e)}")


if __name__ == "__main__":
    main()