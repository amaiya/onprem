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
from OnPrem import DEFAULT_YAML_FPATH, read_config, DEFAULT_YAML
from utils import load_llm, check_manage_access

def main():
    """
    Manage page for configuring the application and ingesting documents
    """
    # No need to hide webapp sidebar item anymore
    
    # Check if manage page should be accessible
    if not check_manage_access():
        st.error("Access to the Manage page has been disabled in the configuration.")
        st.info("To enable access, set 'show_manage: TRUE' in the configuration file.")
        return
    
    st.markdown("""
    <h1 style="
        color: #0068c9;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0068c9;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
    ">
        <span style="font-size: 1.8rem; margin-right: 0.5rem;">⚙️</span> 
        Manage
    </h1>
    """, unsafe_allow_html=True)
    
    # Create tabs for different settings with updated order
    tab1, tab3, tab2 = st.tabs(["Document Ingestion", "Folder Management", "Configuration"])
    
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
        This tool allows you to ingest documents into the vector store.
        You can upload individual files, a ZIP archive containing multiple documents, or spreadsheet data.
        
        **Document Upload Options:**
        - **Individual files**: Upload multiple document files
        - **ZIP archive**: Upload a ZIP file containing multiple documents
        - **Spreadsheet data**: Upload CSV or Excel files where each row becomes a document
        
        **Supported document formats:**
        - PDF (`.pdf`): Adobe Portable Document Format
        - Word (`.docx`): Microsoft Word documents
        - Text (`.txt`): Plain text files
        - HTML (`.html`, `.htm`): Web pages
        - CSV (`.csv`): Comma-separated values
        - Excel (`.xlsx`): Microsoft Excel spreadsheets
        - PowerPoint (`.pptx`): Microsoft PowerPoint presentations
        - Markdown (`.md`): Markdown text files
        - JSON (`.json`): JavaScript Object Notation files
        
        **Spreadsheet ingestion** is ideal for:
        - Survey responses or feedback data
        - Product descriptions with metadata
        - Customer support tickets
        - Research data with participant information
        - Content databases with tags and categories
        
        Advanced options under "Ingestion Options" allow you to:
        - Customize chunk size and overlap
        - Preserve paragraph structure during chunking
        - Control batch processing size
        - Clear existing documents before ingestion
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
            
            
            # Get store type from config
            store_type = cfg.get("llm", {}).get("store_type", "dense")
            
            # Initialize placeholders for file upload UI (will be set later)
            uploaded_files = None
            uploaded_zip = None
            
            # Display preview of selected files based on upload type
            if uploaded_files:
                # Show preview of selected files
                with st.expander(f"Selected {len(uploaded_files)} file(s) for upload", expanded=False):
                    file_info = []
                    total_size = 0
                    
                    # Collect file information
                    for f in uploaded_files:
                        size_kb = f.size / 1024
                        total_size += size_kb
                        file_info.append({
                            "name": f.name, 
                            "type": f.type if hasattr(f, 'type') and f.type else "Unknown",
                            "size": f"{size_kb:.1f} KB"
                        })
                    
                    # Display files in a table
                    st.write("Files to upload:")
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.write("**Filename**")
                    with col2:
                        st.write("**Type**")
                    with col3:
                        st.write("**Size**")
                        
                    for file in file_info:
                        col1, col2, col3 = st.columns([3, 2, 1])
                        with col1:
                            st.write(file["name"])
                        with col2:
                            st.write(file["type"])
                        with col3:
                            st.write(file["size"])
                            
                    st.info(f"Total size: {total_size/1024:.2f} MB")
            
            elif uploaded_zip:
                # Show ZIP file details
                size_mb = uploaded_zip.size / (1024 * 1024)
                st.info(f"Selected ZIP file: {uploaded_zip.name} ({size_mb:.2f} MB)")
                
                # Show preview of ZIP contents if possible
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Save the uploaded zip file to a temporary file
                        temp_zip_path = os.path.join(temp_dir, "preview.zip")
                        with open(temp_zip_path, "wb") as f:
                            f.write(uploaded_zip.getvalue())
                        
                        # Read ZIP contents without extracting
                        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                            zip_files = zip_ref.namelist()
                            
                            # Display file count and preview in expander
                            with st.expander(f"ZIP archive contains {len(zip_files)} file(s)", expanded=False):
                                # Show only first 50 files to avoid UI overload
                                preview_files = zip_files[:50]
                                for file in preview_files:
                                    # Skip directories
                                    if not file.endswith('/'):
                                        st.write(f"📄 {file}")
                                
                                if len(zip_files) > 50:
                                    st.write(f"*...and {len(zip_files) - 50} more files*")
                except Exception as e:
                    st.warning(f"Unable to preview ZIP contents: {str(e)}")
            
            st.markdown('**STEP 1: Select or create a folder in which your uploaded documents will be stored.**')
            st.markdown('*Folders are used to organize your uploaded documents.*')

            # Subfolder selection - require using subfolders
            subfolder_option = st.radio(
                "Upload documents to:",
                ["Existing subfolder", "Create new subfolder"],
                index=0
            )
            
            
            target_folder = None  # Will be set to a valid subfolder path, never the main path
            if subfolder_option == "Existing subfolder":
                # Get list of existing subfolders
                subfolders = [d for d in os.listdir(rag_source_path) 
                             if os.path.isdir(os.path.join(rag_source_path, d))]
                if subfolders:
                    selected_subfolder = st.selectbox("Select subfolder", subfolders)
                    target_folder = os.path.join(rag_source_path, selected_subfolder)
                else:
                    st.info("⚠️ For better organization, documents are uploaded into subfolders. Enter a subfolder name and press ENTER.")
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
            
            # Display selected target folder and validate
            if target_folder and os.path.exists(target_folder):
                relative_path = os.path.relpath(target_folder, rag_source_path)
                st.success(f"Selected upload location: {relative_path}/")
            else:
                st.error("Please select an existing subfolder or create a new one above before uploading.")
            

            st.markdown('**STEP 2: [OPTIONAL] Customize document processing settings.**')
            st.markdown('*We recommend leaving the defaults for most users.*')
            # Place all ingestion options in an expander
            with st.expander("Ingestion Options", expanded=False):
                st.subheader("Document Processing Settings")
                
                # Option to keep full documents (placed first to control other options)
                keep_full_document = st.checkbox("Keep full documents without chunking", value=False,
                                                help="If checked, multi-page documents will be concatenated into single documents with page breaks, and chunking will be disabled. This preserves complete document context but may impact search performance for very long documents.")
                
                # Max words option (only available when keep_full_document is checked)
                max_words = 0
                if keep_full_document:
                    max_words = st.number_input("Maximum words per document", 
                                               min_value=0, 
                                               max_value=1000000, 
                                               value=0,
                                               help="Maximum number of words to extract from each document (0 = no limit)")
                
                # Show warning when keep_full_document is enabled
                if keep_full_document:
                    st.info("ℹ️ Chunking options below are disabled when keeping full documents.")
                
                # Chunking settings
                col1, col2 = st.columns(2)
                with col1:
                    chunk_size = st.number_input("Chunk Size", 
                                               min_value=100, 
                                               max_value=1000000, 
                                               value=500,
                                               help="Text is split into chunks of this many characters",
                                               disabled=keep_full_document)
                with col2:
                    chunk_overlap = st.number_input("Chunk Overlap", 
                                                  min_value=0, 
                                                  max_value=500, 
                                                  value=50,
                                                  help="Character overlap between chunks",
                                                  disabled=keep_full_document)
                
                # Paragraph preservation option
                preserve_paragraphs = st.checkbox(
                    "Preserve paragraphs during chunking", 
                    value=False,
                    help="If checked, documents will be chunked at paragraph boundaries. If a paragraph exceeds the chunk size, it will be split. If unchecked, small paragraphs will be combined into a single chunk until the chunk size is reached. We recommend you leave this off unless you are performing a **Document Analysis** that would benefit from retaining paragraphs.",
                    disabled=keep_full_document
                )
                
                # Display store type (read-only)
                st.info(f"Using store type from configuration: {store_type}")
                
                # Batch size
                batch_size = st.number_input("Batch Size", 
                                            min_value=1, 
                                            max_value=5000, 
                                            value=1000,
                                            help="Number of documents to process in each batch")
                
                # Option to infer table structure
                infer_table_structure = st.checkbox("Infer table structure from documents", value=False,
                                                  help="If checked, attempts to detect and preserve table structures when processing documents")
                
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
            
            # Folder management expandable section
            with tab3:
                st.subheader("Folder Management")
                
                # Add refresh button
                if st.button("Refresh Folder List", key="refresh_folders"):
                    st.rerun()
                
                # Get list of existing subfolders
                subfolders = [d for d in os.listdir(rag_source_path) 
                             if os.path.isdir(os.path.join(rag_source_path, d))]
                
                # Display folder statistics
                if subfolders:
                    st.write(f"Found {len(subfolders)} subfolder(s) in document root directory:")
                    
                    # Create columns for folder stats
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write("**Folder Name**")
                    with col2:
                        st.write("**File Count**")
                    
                    # Display each folder with file count
                    for folder in subfolders:
                        folder_path = os.path.join(rag_source_path, folder)
                        file_count = sum([len(files) for _, _, files in os.walk(folder_path)])
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"📁 {folder}")
                        with col2:
                            st.write(f"{file_count}")
                    
                    st.write("---")
                    
                
                if not subfolders:
                    st.info("No subfolders exist yet. You can create subfolders under the Document Ingestion tab.")
                else:
                    st.write("Select a subfolder to delete:")
                    folder_to_delete = st.selectbox("Subfolder to delete", 
                                                  subfolders,
                                                  key="delete_folder_select")
                    
                    # Two-step deletion with session state
                    folder_path = os.path.join(rag_source_path, folder_to_delete)
                    
                    # Count files that will be deleted
                    file_count = sum([len(files) for _, _, files in os.walk(folder_path)])
                    
                    # Show folder contents
                    st.write(f"Contents of '{folder_to_delete}' folder:")
                    items = os.listdir(folder_path)
                    if items:
                        for item in items[:10]:  # Limit to first 10 items to avoid clutter
                            item_path = os.path.join(folder_path, item)
                            if os.path.isdir(item_path):
                                st.markdown(f"📁 **{item}/** *(folder)*")
                            else:
                                st.markdown(f"📄 **{item}**")
                        if len(items) > 10:
                            st.markdown(f"*...and {len(items) - 10} more items*")
                    else:
                        st.info("This folder is empty.")
                    
                    # Use a different approach without modifying widget keys
                    # Create a session state variable for the delete process
                    if "delete_state" not in st.session_state:
                        st.session_state.delete_state = {"stage": 0, "folder": None}
                    
                    # Store currently selected folder for deletion
                    current_folder = folder_to_delete
                    
                    # Handle the deletion process based on current state
                    if st.session_state.delete_state["stage"] == 0:
                        # Initial stage - show delete button
                        if st.button("Delete this folder"):
                            # Move to confirmation stage and store folder name
                            st.session_state.delete_state = {"stage": 1, "folder": current_folder}
                            st.rerun()
                    
                    elif st.session_state.delete_state["stage"] == 1:
                        if st.session_state.delete_state["folder"] == current_folder:
                            # Confirmation stage
                            st.warning(f"⚠️ You are about to permanently delete folder '{current_folder}' and all its contents ({file_count} files)!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Cancel"):
                                    # Reset to initial stage
                                    st.session_state.delete_state = {"stage": 0, "folder": None}
                                    st.rerun()
                            with col2:
                                if st.button("Confirm Delete"):
                                    try:
                                        # Remove documents from vectorstore
                                        st.text(f"Removing documents from '{current_folder}' in vectorstore...")
                                            
                                            
                                        # Load LLM and get vectorstore
                                        llm = load_llm()
                                        vectorstore = llm.load_vectorstore()
                                       
                                        # Delete documents from vectorstore
                                        normed_folder_path = os.path.normpath(folder_path).replace('\\', '/')
                                        num_records_deleted = vectorstore.remove_source(normed_folder_path)
                                        st.text(f'{num_records_deleted} records deleted from vectorstore.')
                                        
                                        # Delete the folder
                                        shutil.rmtree(folder_path)
                                        st.success(f"Subfolder '{current_folder}' has been deleted successfully.")

                                        # Reset state and refresh
                                        st.session_state.delete_state = {"stage": 0, "folder": None}
                                        
                                        # Create a countdown before refresh
                                        import time
                                        countdown_seconds = 5
                                        countdown_placeholder = st.empty()
                                        for i in range(countdown_seconds, 0, -1):
                                            countdown_placeholder.info(f"Refreshing in {i} seconds...")
                                            time.sleep(1)
                                        countdown_placeholder.info("Refreshing now...")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error deleting folder: {str(e)}")
                        else:
                            # Selected folder changed, reset state
                            st.session_state.delete_state = {"stage": 0, "folder": None}
                            st.rerun()
            
            # Ingest button is placed outside all expanders for visibility


            st.markdown('**STEP 3: Select and upload files for ingestion into the document store.**')
            st.info(f"Documents store location: {rag_source_path}")
            # Create columns to place file uploader next to the ingest button
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Create a radio selector to choose file upload type
                upload_type = st.radio(
                    "What would you like to upload?",
                    ["Individual files", "ZIP archive", "Spreadsheet data"],
                    horizontal=True
                )
                
                # Move file uploaders here
                if upload_type == "Individual files":
                    # Individual file uploader with multiple file support
                    uploaded_files = st.file_uploader(
                        "Upload individual document files", 
                        accept_multiple_files=True,
                        type=["pdf", "docx", "txt", "csv", "html", "htm", "md", "json", "xlsx", "pptx"]
                    )
                    uploaded_zip = None
                    uploaded_spreadsheet = None
                elif upload_type == "ZIP archive":
                    # ZIP file upload
                    uploaded_zip = st.file_uploader("Upload ZIP file containing multiple documents", type="zip")
                    uploaded_files = None
                    uploaded_spreadsheet = None
                else:
                    # Spreadsheet data upload
                    uploaded_spreadsheet = st.file_uploader(
                        "Upload spreadsheet file (CSV, Excel)", 
                        type=["csv", "xlsx", "xls"]
                    )
                    uploaded_files = None
                    uploaded_zip = None
                    
                    # Initialize variables for spreadsheet configuration
                    text_column = "text"
                    sheet_name = None
                    metadata_columns = ""
                    
                    # Spreadsheet configuration
                    if uploaded_spreadsheet:
                        st.markdown("**Spreadsheet Configuration:**")
                        col1_config, col2_config = st.columns(2)
                        
                        with col1_config:
                            text_column = st.text_input(
                                "Text Column Name", 
                                value="text", 
                                help="Name of column containing document text content"
                            )
                        
                        with col2_config:
                            if uploaded_spreadsheet.name.endswith(('.xlsx', '.xls')):
                                sheet_name = st.text_input(
                                    "Sheet Name (Excel only)", 
                                    value="", 
                                    help="Leave empty for first sheet"
                                )
                                if not sheet_name.strip():
                                    sheet_name = None
                            else:
                                sheet_name = None
                        
                        # Metadata columns configuration
                        metadata_columns = st.text_input(
                            "Metadata Columns (optional)",
                            value="",
                            help="Comma-separated column names to include as metadata. Leave empty to use all columns except text column."
                        )
                        
                        # Show preview if possible
                        try:
                            import pandas as pd
                            import io
                            
                            # Create a copy of the file content for preview
                            file_content = uploaded_spreadsheet.getvalue()
                            
                            # Read a preview of the spreadsheet
                            if uploaded_spreadsheet.name.endswith('.csv'):
                                df_preview = pd.read_csv(io.BytesIO(file_content), nrows=5)
                            else:
                                df_preview = pd.read_excel(io.BytesIO(file_content), nrows=5, sheet_name=sheet_name if sheet_name else 0)
                            
                            with st.expander("Spreadsheet Preview (first 5 rows)", expanded=False):
                                st.dataframe(df_preview)
                                st.info(f"Available columns: {', '.join(df_preview.columns.tolist())}")
                                
                        except Exception as e:
                            st.warning(f"Could not preview spreadsheet: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            
            # Ingest button
            has_files_to_upload = (uploaded_files and len(uploaded_files) > 0) or uploaded_zip is not None or uploaded_spreadsheet is not None
            has_valid_subfolder = target_folder and os.path.exists(target_folder) and target_folder != rag_source_path
            
            upload_button = st.button("Upload and Ingest Documents")

            if upload_button and not has_valid_subfolder:
                st.error('Please select a subfolder above.')
            elif upload_button and not has_files_to_upload:
                st.error('Please drag and drop files to upload.')
            elif has_files_to_upload and has_valid_subfolder and upload_button:
                try:
                    with st.spinner("Processing and ingesting documents..."):
                        # Create target folder if it doesn't exist
                        os.makedirs(target_folder, exist_ok=True)
                        
                        # Clear the target directory if requested
                        if clear_existing and os.path.exists(target_folder):
                            folder_display = os.path.basename(target_folder) if target_folder != rag_source_path else "main folder"
                            st.text(f"Clearing existing documents from {folder_display}...")
                            for item in os.listdir(target_folder):
                                item_path = os.path.join(target_folder, item)
                                if os.path.isfile(item_path):
                                    os.remove(item_path)
                                elif os.path.isdir(item_path):
                                    shutil.rmtree(item_path)
                        
                        # Create a temporary directory for processing files
                        with tempfile.TemporaryDirectory() as temp_dir:
                            processed_files = 0
                            
                            # Handle individual files if available
                            if uploaded_files and len(uploaded_files) > 0:
                                st.text(f"Saving {len(uploaded_files)} individual files...")
                                
                                for uploaded_file in uploaded_files:
                                    # Create a safe filename (remove special characters)
                                    safe_filename = ''.join(c for c in uploaded_file.name if c.isalnum() or c in '._- ')
                                    
                                    # Save the file to the target directory
                                    file_path = os.path.join(target_folder, safe_filename)
                                    with open(file_path, "wb") as f:
                                        f.write(uploaded_file.getvalue())
                                    
                                    processed_files += 1
                            
                            # Handle ZIP file if available
                            if uploaded_zip is not None:
                                st.text("Processing ZIP archive...")
                                
                                # Save the uploaded zip file to a temporary file
                                temp_zip_path = os.path.join(temp_dir, "uploaded.zip")
                                with open(temp_zip_path, "wb") as f:
                                    f.write(uploaded_zip.getvalue())
                                
                                # Extract the zip file to the target directory
                                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                                    # Get list of files in the zip
                                    zip_files = zip_ref.namelist()
                                    print(zip_files)
                                    st.text(f"Extracting {len(zip_files)} files/folders from ZIP archive...")
                                    zip_ref.extractall(target_folder)
                                    processed_files += len(zip_files)
                            
                            # Handle spreadsheet if available
                            if uploaded_spreadsheet is not None:
                                st.text("Processing spreadsheet data...")
                                
                                # Parse metadata columns
                                metadata_cols = None
                                if metadata_columns.strip():
                                    metadata_cols = [col.strip() for col in metadata_columns.split(',') if col.strip()]
                                
                                # Save spreadsheet to temporary file for processing
                                temp_file_path = os.path.join(temp_dir, uploaded_spreadsheet.name)
                                with open(temp_file_path, "wb") as f:
                                    f.write(uploaded_spreadsheet.getvalue())
                                
                                # Also save the spreadsheet to the target folder for reference
                                spreadsheet_path = os.path.join(target_folder, uploaded_spreadsheet.name)
                                with open(spreadsheet_path, "wb") as f:
                                    f.write(uploaded_spreadsheet.getvalue())
                                
                                # Use load_spreadsheet_documents to process the file
                                from onprem.ingest.base import load_spreadsheet_documents
                                
                                try:
                                    st.text(f"Loading spreadsheet with text_column='{text_column}', metadata_columns={metadata_cols}, sheet_name='{sheet_name}'")
                                    
                                    documents = load_spreadsheet_documents(
                                        file_path=temp_file_path,
                                        text_column=text_column,
                                        metadata_columns=metadata_cols,
                                        sheet_name=sheet_name
                                    )
                                    
                                    st.text(f"Loaded {len(documents)} documents from spreadsheet")
                                    
                                    # Process and chunk documents
                                    from onprem.ingest.base import chunk_documents
                                    
                                    # Apply chunking and preservation settings
                                    chunked_documents = chunk_documents(
                                        documents=documents,
                                        chunk_size=chunk_size,
                                        chunk_overlap=chunk_overlap,
                                        preserve_paragraphs=preserve_paragraphs,
                                        infer_table_structure=infer_table_structure
                                    )
                                    
                                    st.text(f"Created {len(chunked_documents)} document chunks")
                                    
                                    # Debug: Show sample metadata
                                    if chunked_documents:
                                        sample_doc = chunked_documents[0]
                                        st.text(f"Sample document metadata keys: {list(sample_doc.metadata.keys())}")
                                    
                                    processed_files = len(documents)
                                    
                                    # Load LLM and ingest directly using vector store
                                    llm = load_llm()
                                    
                                    # Get the vector store instance  
                                    vectorstore = llm.load_vectorstore()
                                    
                                    # Clean up documents by removing None values from metadata and fixing source path
                                    from langchain_core.documents import Document
                                    cleaned_documents = []
                                    for doc in chunked_documents:
                                        # Remove None values from metadata
                                        cleaned_metadata = {k: v for k, v in doc.metadata.items() if v is not None}
                                        
                                        # Update the source to point to the uploaded spreadsheet file in the target folder
                                        spreadsheet_path = os.path.join(target_folder, uploaded_spreadsheet.name)
                                        cleaned_metadata['source'] = spreadsheet_path
                                        # Also update meta_source if it exists
                                        if 'meta_source' in cleaned_metadata:
                                            cleaned_metadata['meta_source'] = spreadsheet_path
                                        
                                        cleaned_doc = Document(
                                            page_content=doc.page_content,
                                            metadata=cleaned_metadata
                                        )
                                        cleaned_documents.append(cleaned_doc)
                                    
                                    # Add documents directly to the vector store
                                    try:
                                        vectorstore.add_documents(cleaned_documents)
                                    except Exception as e:
                                        st.error(f"Vector store ingestion error: {str(e)}")
                                        # Try to show more details about the problematic metadata
                                        if cleaned_documents:
                                            st.text("Sample cleaned document metadata:")
                                            sample_metadata = cleaned_documents[0].metadata
                                            for key, value in sample_metadata.items():
                                                st.text(f"  {key}: {type(value).__name__} = {str(value)[:100]}")
                                        raise
                                    
                                    # Create result dict similar to llm.ingest output
                                    result = {"num_added": len(chunked_documents)}
                                    
                                    # Show success message
                                    folder_display = "spreadsheet data"
                                    st.success(f"Successfully processed and ingested {folder_display}")
                                    
                                    # Show stats if available
                                    if isinstance(result, dict) and "num_added" in result:
                                        st.info(f"Added {result['num_added']} new document chunks to the vector database")
                                    
                                    return  # Exit early since we processed directly
                                    
                                except Exception as e:
                                    st.error(f"Error processing spreadsheet: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())
                                    return
                            
                            folder_display_name = os.path.basename(target_folder) if target_folder != rag_source_path else "main folder"
                            st.text(f"Saved to {folder_display_name} subfolder ({processed_files} files/folders processed)")
                            
                            # Now ingest the documents
                            st.text("Ingesting documents...")
                            llm = load_llm()
                            
                            # Ingest documents from the target folder
                            # Set n_proc=1 on Windows to avoid multiprocessing issues with Streamlit
                            kwargs = {
                                "source_directory": target_folder,
                                "chunk_size": chunk_size,
                                "chunk_overlap": chunk_overlap,
                                "batch_size": batch_size,
                                "preserve_paragraphs": preserve_paragraphs,
                                "infer_table_structure": infer_table_structure,
                                "keep_full_document": keep_full_document,
                                "max_words": max_words if max_words > 0 else None
                            }
                            # Add n_proc=1 on Windows to avoid multiprocessing stalls
                            if os.name == 'nt':  # Windows
                                kwargs["n_proc"] = 1
                                st.text("Windows detected: Using single process mode for document ingestion")
                            
                            result = llm.ingest(**kwargs)
                            
                            # Show success message with folder information
                            folder_display = os.path.basename(target_folder) if target_folder != rag_source_path else "main folder"
                            st.success(f"Successfully uploaded and ingested documents to {folder_display}")
                            
                            # Show stats if available
                            if isinstance(result, dict) and "num_added" in result:
                                st.info(f"Added {result['num_added']} new document chunks to the vector database")
                except Exception as e:
                    st.error(f"Error during document upload and ingestion: {str(e)}")
                    #import traceback
                    #st.error(traceback.format_exc())
                    

if __name__ == "__main__":
    # Set page to wide mode when run directly
    st.set_page_config(
        page_title="Manage", 
        page_icon="⚙️", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    main()
