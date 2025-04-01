import os
import sys
import yaml
import streamlit as st
from pathlib import Path
import zipfile
import tempfile
import shutil
import re

# Add parent directory to path to allow importing when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from parent modules
from webapp import DEFAULT_YAML_FPATH, read_config, DEFAULT_YAML
from utils import hide_webapp_sidebar_item, load_llm, check_manage_access

def main():
    """
    Manage page for configuring the application and ingesting documents
    """
    # Hide webapp sidebar item
    hide_webapp_sidebar_item()
    
    # Check if manage page should be accessible
    if not check_manage_access():
        st.error("Access to the Manage page has been disabled in the configuration.")
        st.info("To enable access, set 'show_manage: TRUE' in the configuration file.")
        return
    
    st.header("Manage")
    
    # Create tabs for different settings with updated order
    tab1, tab2 = st.tabs(["Document Ingestion", "Configuration"])
    
    with tab2:
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
    
    with tab1:
        st.subheader("Document Ingestion")
        
        # Instructions
        st.markdown("""
        This tool allows you to ingest documents into the vector database.
        You can upload a ZIP file containing documents, and they will be automatically extracted and ingested.
        
        Supported formats include PDF, DOCX, TXT, HTML, CSV, and more.
        """)
        
        # Get config to extract rag_source_path
        cfg = read_config()[0]
        rag_source_path = cfg.get("ui", {}).get("rag_source_path", "")
        
        if not rag_source_path:
            st.error("No document source path configured. Please set the 'rag_source_path' value in the Configuration tab first.")
        else:
            # Create the directory if it doesn't exist
            from onprem import utils as U
            
            # Replace any placeholders in the path
            rag_source_path = rag_source_path.format(webapp_dir=U.get_webapp_dir())
            os.makedirs(rag_source_path, exist_ok=True)
            
            st.info(f"Documents root directory: {rag_source_path}")
            
            # Get store type from config
            store_type = cfg.get("llm", {}).get("store_type", "dense")
            
            # ZIP file upload - kept outside the expander
            uploaded_file = st.file_uploader("Upload ZIP file containing documents", type="zip")
            
            # Subfolder selection
            subfolder_option = st.radio(
                "Upload documents to:",
                ["Main folder", "Existing subfolder", "Create new subfolder"],
                index=0
            )
            
            target_folder = rag_source_path
            if subfolder_option == "Existing subfolder":
                # Get list of existing subfolders
                subfolders = [d for d in os.listdir(rag_source_path) 
                             if os.path.isdir(os.path.join(rag_source_path, d))]
                if subfolders:
                    selected_subfolder = st.selectbox("Select subfolder", subfolders)
                    target_folder = os.path.join(rag_source_path, selected_subfolder)
                else:
                    st.warning("No existing subfolders found. Creating one first.")
                    subfolder_option = "Create new subfolder"
            
            if subfolder_option == "Create new subfolder":
                new_subfolder = st.text_input("Enter subfolder name")
                if new_subfolder:
                    # Validate folder name (no special characters except underscore, hyphen)
                    if not re.match(r'^[a-zA-Z0-9_\-]+$', new_subfolder):
                        st.error("Folder name can only contain letters, numbers, underscores, and hyphens.")
                    else:
                        target_folder = os.path.join(rag_source_path, new_subfolder)
                        if not os.path.exists(target_folder):
                            os.makedirs(target_folder, exist_ok=True)
                            st.success(f"Created subfolder: {new_subfolder}")
            
            # Display selected target folder
            if target_folder != rag_source_path:
                relative_path = os.path.relpath(target_folder, rag_source_path)
                st.info(f"Selected upload location: {relative_path}/")
            else:
                st.info("Selected upload location: Main folder")
            
            # Place all ingestion options in an expander
            with st.expander("Ingestion Options", expanded=False):
                st.subheader("Document Processing Settings")
                
                # Chunking settings
                col1, col2 = st.columns(2)
                with col1:
                    chunk_size = st.number_input("Chunk Size", 
                                               min_value=100, 
                                               max_value=1000000, 
                                               value=500,
                                               help="Text is split into chunks of this many characters")
                with col2:
                    chunk_overlap = st.number_input("Chunk Overlap", 
                                                  min_value=0, 
                                                  max_value=500, 
                                                  value=50,
                                                  help="Character overlap between chunks")
                
                # Display store type (read-only)
                st.info(f"Using store type from configuration: {store_type}")
                
                # Batch size
                batch_size = st.number_input("Batch Size", 
                                            min_value=1, 
                                            max_value=5000, 
                                            value=1000,
                                            help="Number of documents to process in each batch")
                
                # Option to clear existing documents
                clear_existing = st.checkbox("Clear existing documents before ingestion", value=False, 
                                             help="If checked, all existing documents in the target directory will be removed before extracting new ones")
                
                # If clear_existing is checked, show the current contents of the target directory
                if clear_existing:
                    if os.path.exists(target_folder):
                        items = os.listdir(target_folder)
                        folder_display = os.path.basename(target_folder) if target_folder != rag_source_path else "main folder"
                        
                        if items:
                            st.warning(f"The following files and folders will be deleted from {folder_display}:")
                            for item in items:
                                item_path = os.path.join(target_folder, item)
                                if os.path.isdir(item_path):
                                    st.markdown(f"📁 **{item}/** *(folder)*")
                                else:
                                    st.markdown(f"📄 **{item}**")
                        else:
                            st.info(f"The selected folder ({folder_display}) is currently empty.")
                    else:
                        st.info("The selected folder does not exist yet and will be created.")
            
            # Ingest button is placed outside the expander for visibility


            # Ingest button
            if uploaded_file is not None and st.button("Upload and Ingest Documents"):
                try:
                    with st.spinner("Extracting and ingesting documents..."):
                        # Create a temporary directory to extract the zip file
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Save the uploaded zip file to a temporary file
                            temp_zip_path = os.path.join(temp_dir, "uploaded.zip")
                            with open(temp_zip_path, "wb") as f:
                                f.write(uploaded_file.getvalue())
                            
                            # Create target folder if it doesn't exist
                            os.makedirs(target_folder, exist_ok=True)
                            
                            # Clear the target directory if requested
                            if clear_existing and os.path.exists(target_folder):
                                st.text(f"Clearing existing documents from {os.path.basename(target_folder) if target_folder != rag_source_path else 'main folder'}...")
                                for item in os.listdir(target_folder):
                                    item_path = os.path.join(target_folder, item)
                                    if os.path.isfile(item_path):
                                        os.remove(item_path)
                                    elif os.path.isdir(item_path):
                                        shutil.rmtree(item_path)
                            
                            # Extract the zip file to the target directory
                            st.text("Extracting ZIP file...")
                            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                                zip_ref.extractall(target_folder)
                            
                            folder_display_name = os.path.basename(target_folder) if target_folder != rag_source_path else "main folder"
                            st.text(f"Files extracted to {folder_display_name}")
                            
                            # Now ingest the documents
                            st.text("Ingesting documents...")
                            llm = load_llm()
                            
                            # Ingest documents from the target folder where we extracted files
                            result = llm.ingest(
                                source_directory=target_folder,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                batch_size=batch_size
                            )
                            
                            # Show success message with folder information
                            folder_display = os.path.basename(target_folder) if target_folder != rag_source_path else "main folder"
                            st.success(f"Successfully uploaded and ingested documents to {folder_display}")
                            
                            # Show stats if available
                            if isinstance(result, dict) and "num_added" in result:
                                st.info(f"Added {result['num_added']} new document chunks to the vector database")
                except Exception as e:
                    st.error(f"Error during document upload and ingestion: {str(e)}")
                    
            ## Alternative manual folder ingest (for advanced users)
            #with st.expander("Advanced: Ingest from existing folder"):
                #st.markdown("""
                #For advanced users: If you already have documents in a folder on the server, 
                #you can ingest them directly by providing the folder path below.
                #""")
                
                ## Folder selection
                #folder_path = st.text_input("Folder Path", 
                                           #placeholder="Enter the absolute path to your documents folder",
                                           #help="Enter the full path to the folder containing documents to ingest")
                
                ## Ingest button for manual folder
                #if st.button("Ingest from Folder"):
                    #if not folder_path:
                        #st.error("Please enter a valid folder path")
                    #elif not os.path.isdir(folder_path):
                        #st.error(f"Directory does not exist: {folder_path}")
                    #else:
                        #try:
                            #with st.spinner(f"Ingesting documents from {folder_path}..."):
                                ## Load the LLM with the current configuration
                                #llm = load_llm()
                                
                                ## Ingest documents
                                #result = llm.ingest(
                                    #source_directory=folder_path,
                                    #chunk_size=chunk_size,
                                    #chunk_overlap=chunk_overlap,
                                    #batch_size=batch_size
                                #)
                                
                                ## Show success message
                                #st.success(f"Successfully ingested documents from {folder_path}")
                                
                                ## Show stats if available
                                #if isinstance(result, dict) and "num_added" in result:
                                    #st.info(f"Added {result['num_added']} new document chunks to the vector database")
                        #except Exception as e:
                            #st.error(f"Error during document ingestion: {str(e)}")


if __name__ == "__main__":
    main()