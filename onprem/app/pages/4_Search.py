import os
import sys
import streamlit as st
from typing import Type, Optional, Tuple
import mimetypes

# Add parent directory to path to allow importing when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from parent modules
from webapp import read_config
from utils import hide_webapp_sidebar_item, construct_link, check_create_symlink
from utils import load_llm, lucene_to_chroma
from onprem.ingest.stores.sparse import SparseStore
from onprem.ingest.stores.dense import DenseStore
from onprem.ingest.stores.dual import DualStore

# Helper function to get file data for download
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

def main():
    """
    Page for searching documents using sparse vector store
    """
    # Hide webapp sidebar item
    hide_webapp_sidebar_item()
    
    cfg = read_config()[0]
    
    # Get configuration
    RAG_SOURCE_PATH = cfg.get("ui", {}).get("rag_source_path", None)
    RAG_BASE_URL = cfg.get("ui", {}).get("rag_base_url", None)
    VECTORDB_PATH = cfg.get("llm", {}).get("vectordb_path", None)
    STORE_TYPE = cfg.get("llm", {}).get("store_type", "dense")
    
    # Load LLM to get model name
    llm = load_llm()
    MODEL_NAME = llm.model_name
    RAG_SOURCE_PATH, RAG_BASE_URL = check_create_symlink(RAG_SOURCE_PATH, RAG_BASE_URL)
    
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
    st.markdown("""
    <h1 style="
        color: #0068c9;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0068c9;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
    ">
        <span style="font-size: 1.8rem; margin-right: 0.5rem;">🔍</span> 
        Document Search
    </h1>
    """, unsafe_allow_html=True)
    
    # Enhanced info card - same style as in Prompts.py
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
        <span style="font-size: 1.2rem; margin-right: 10px;">🔍</span>
        <div>
            <p style="margin: 0; font-weight: 500;">Search Type: <span style="color: #0068c9;">{STORE_TYPE.capitalize()} Search</span></p>
            <p style="margin: 3px 0 0 0; font-size: 0.85rem; color: #666;">
                Semantic search finds conceptually similar documents. Keyword search finds exact matches.
            </p>
        </div>
    </div>
    """  , unsafe_allow_html=True)
    
    st.markdown("""
    Search through your indexed documents using keywords and filters.
    """)
    
    # Check if vector database exists
    if not VECTORDB_PATH or not os.path.exists(VECTORDB_PATH):
        st.error("No vector database found. Please ingest documents first by going to Manage.")
        return
    
    # Initialize the vector store using LLM.load_vectorstore()
    try:
        
        # Load the vector store
        vectorstore = llm.load_vectorstore()
        store_type = llm.get_store_type()
        
        # Check if store exists and has documents
        if not vectorstore.exists():
            st.error("No documents have been indexed. Please ingest documents first.")
            return
            
        # Determine if keyword search is available
        # Only allow keyword search if vectorstore is NOT a DenseStore (i.e., it's a SparseStore or DualStore)
        has_keyword_search = not isinstance(vectorstore, DenseStore)
            
    except Exception as e:
        st.error(f"Error loading search index: {str(e)}")
        return
    
    # Initialize session state for search terms and filters if not already done
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    if 'search_type' not in st.session_state:
        # Default to "Semantic" if store_type is "dense", otherwise "Keyword"
        st.session_state.search_type = "Semantic" if STORE_TYPE == "dense" else "Keyword"
    if 'where_document' not in st.session_state:
        st.session_state.where_document = ""
    if 'results_limit' not in st.session_state:
        st.session_state.results_limit = 5000 
    if 'deduplicate_sources' not in st.session_state:
        st.session_state.deduplicate_sources = False
    # Pagination parameters
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    if 'results_per_page' not in st.session_state:
        st.session_state.results_per_page = 5
    
    # Create search interface
    col1, col2 = st.columns([3, 1])
    
    # Update session state when search terms change
    def on_query_change():
        st.session_state.search_query = st.session_state.query_input
        
    with col1:
        query = st.text_input("Enter search terms:", 
                              placeholder="Enter keywords to search for",
                              value=st.session_state.search_query,
                              key="query_input",
                              on_change=on_query_change)
    
    with col2:
        # Only show "Keyword" option if the vectorstore supports it
        search_options = ["Semantic"]
        if has_keyword_search:
            search_options.insert(0, "Keyword")
        
        # Reset search type if it was set to keyword but keyword search is no longer available
        if st.session_state.search_type == "Keyword" and not has_keyword_search:
            st.session_state.search_type = "Semantic"
            
        default_index = 0
        if "Keyword" in search_options and st.session_state.search_type == "Keyword":
            default_index = 0
        else:
            default_index = search_options.index("Semantic")
            
        search_type = st.selectbox("Search type:", 
                                  search_options, 
                                  index=default_index)
    
    # Filters section (collapsible)
    with st.expander("Advanced Filters and Search Settings"):
        filter_options = {} # not currently implemented
        
        # Custom where clause
        where_document = st.text_input("Custom query filter:", 
                                      placeholder='e.g., "ChatGPT" AND extension:pdf',
                                      help="Use AND, OR, NOT operators for complex queries",
                                      value=st.session_state.where_document)
        
        # Search settings
        st.subheader("Search Settings")
        st.info('💡 OnPrem splits documents into text "chunks" during ingestion.')
        results_limit = st.slider("Maximum number of chunks to retrieve in search:", 
                                min_value=1000, 
                                max_value=25000, 
                                value=st.session_state.results_limit, 
                                step=1000)
        
        # Pagination settings
        results_per_page = st.select_slider(
            "Results per page:",
            options=[5, 10, 20, 50, 100],
            value=st.session_state.results_per_page
        )
        
        # De-duplication option
        deduplicate_sources = st.checkbox(
            "Collapse chunks by document source in search results",
            value=st.session_state.deduplicate_sources,
            help="If checked, results are de-duplicated by document source, with chunk content from all matches concatenated"
        )
    
    # Search and reset buttons side by side
    button_cols = st.columns([1, 1, 4])  # First two columns for buttons, third for spacing
    with button_cols[0]:
        search_button = st.button("Search", type="primary", use_container_width=True)
    with button_cols[1]:
        # Place the reset button immediately to the right of the search button
        reset_button = st.button("Reset", type="secondary", use_container_width=True)
    
    # Reset search state if reset button is clicked
    if reset_button:
        st.session_state.search_query = ""
        # Set default search type based on available search types
        st.session_state.search_type = "Keyword" if has_keyword_search else "Semantic"
        st.session_state.where_document = ""
        st.session_state.results_limit = 5000 
        st.session_state.deduplicate_sources = False
        # Reset pagination
        st.session_state.current_page = 1
        st.session_state.results_per_page = 5
        st.rerun()
        
    # Update session state when search button is clicked
    if search_button and st.session_state.search_query:
        st.session_state.search_type = search_type
        st.session_state.where_document = where_document
        st.session_state.results_limit = results_limit
        st.session_state.deduplicate_sources = deduplicate_sources
        st.session_state.results_per_page = results_per_page
        # Reset to first page when performing a new search
        st.session_state.current_page = 1
    elif search_button and not st.session_state.search_query:
        st.error("You didn't enter a search.")
        st.stop()

    # Handle search - execute search if we have a query in session state
    if st.session_state.search_query:
        with st.spinner("Searching..."):
            try:
                # Verify search type is valid for the current vectorstore
                if st.session_state.search_type == "Keyword" and not has_keyword_search:
                    st.error("Keyword search is not available with the current vector store configuration.")
                    return
                
                if st.session_state.search_type == "Keyword":
                    # Use query method of vectorstore for keyword search
                    results = vectorstore.query(
                        q=st.session_state.search_query,
                        limit=st.session_state.results_limit,
                        filters=filter_options,
                        where_document=st.session_state.where_document if st.session_state.where_document else None,
                        highlight=True
                    )
                    hits = results.get('hits', [])
                    total_hits = results.get('total_hits', 0)
                else:  # Semantic search
                    # Use semantic_search method of vectorstore
                    if store_type != 'sparse':
                        # transform Whoosh query filter to Chroma syntax
                        where_document = st.session_state.where_document if st.session_state.where_document else None
                        chroma_filters = lucene_to_chroma(where_document)
                        where_document = chroma_filters['where_document']
                        filter_options = chroma_filters['filter']
                    hits = vectorstore.semantic_search(
                        query=st.session_state.search_query,
                        k=st.session_state.results_limit,
                        filters=filter_options if filter_options else None,
                        where_document=where_document if where_document else None
                    )
                    total_hits = len(hits)
                
                # Handle de-duplication if enabled
                if st.session_state.deduplicate_sources:
                    # Group results by source
                    source_grouped_hits = {}
                    for doc in hits:
                        source = doc.metadata.get("source", "Unknown source")
                        if source not in source_grouped_hits:
                            source_grouped_hits[source] = {
                                "metadata": doc.metadata.copy(),
                                "docs": [],
                                "combined_content": [],
                                "highlights": []
                            }
                        source_grouped_hits[source]["docs"].append(doc)
                        source_grouped_hits[source]["combined_content"].append(doc.page_content)
                        
                        # Capture highlights if available - store up to 3 highlights
                        if st.session_state.search_type == "Keyword" and 'hl_page_content' in doc.metadata and len(source_grouped_hits[source]["highlights"]) < 3:
                            source_grouped_hits[source]["highlights"].append(doc.metadata.get('hl_page_content', ''))
                    
                    # Create a new list of consolidated documents
                    consolidated_hits = []
                    for source, data in source_grouped_hits.items():
                        # Create a new document with combined content
                        consolidated_doc = type('obj', (object,), {
                            'page_content': "\n\n---\n\n".join(data["combined_content"]),
                            'metadata': data["metadata"]
                        })
                        
                            # Add highlight information if available - join up to 3 highlights with separators
                        if data["highlights"]:
                            # Join the highlights (up to 3) with separators
                            consolidated_doc.metadata['hl_page_content'] = "\n• • • • •\n".join(data["highlights"])
                            
                        # Store original count
                        consolidated_doc.metadata['chunk_count'] = len(data["docs"])
                        
                        consolidated_hits.append(consolidated_doc)
                    
                    # Replace the original hits with consolidated ones
                    hits = consolidated_hits
                    original_total_hits = total_hits  # Save the original total for reference
                    total_hits = len(hits)  # Update total_hits to be the consolidated count
                    
                    # Display the result counts - we'll add pagination info after we calculate it
                    st.subheader(f"Search Results: {total_hits} sources found (from {original_total_hits} chunks)")
                else:
                    # Display regular results count - we'll add pagination info after we calculate it
                    st.subheader(f"Search Results: {total_hits} documents found")
                
                if not hits:
                    st.info("No documents found matching your search criteria.")
                    return
                
                # Calculate pagination - this is the single, unified pagination calculation
                total_pages = (total_hits + st.session_state.results_per_page - 1) // st.session_state.results_per_page
                
                # Ensure current_page is valid
                if st.session_state.current_page < 1:
                    st.session_state.current_page = 1
                if st.session_state.current_page > total_pages:
                    st.session_state.current_page = total_pages
                
                # Calculate start and end indices for current page
                start_idx = (st.session_state.current_page - 1) * st.session_state.results_per_page
                end_idx = min(start_idx + st.session_state.results_per_page, total_hits)
                
                # Calculate the item range being displayed for the header
                start_item = start_idx + 1
                end_item = min(start_idx + st.session_state.results_per_page, total_hits)
                
                # Update the header to include pagination information
                st.markdown(f"Showing results {start_item}-{end_item} of {total_hits}")
                
                # Get current page results
                current_page_hits = hits[start_idx:end_idx]
                
                # Display each result for the current page
                for idx, doc in enumerate(current_page_hits):
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        
                        source = doc.metadata.get("source", "Unknown source")
                        # Get page if available - only show for PDFs
                        page = doc.metadata.get("page")
                        extension = doc.metadata.get("extension", "").lower()
                        page_info = ""
                        if page is not None and extension == "pdf":
                            page_info = f", page {page + 1}"
                            
                        is_windows = os.name == 'nt'
                        
                        with col1:
                            # For Windows, use a plain filename + download button
                            # For Linux/macOS, use the standard hyperlink
                            if is_windows:
                                filename = os.path.basename(source)
                                # Display the absolute position in search results
                                result_number = start_idx + idx + 1
                                st.markdown(f"**{result_number}. {filename}{page_info}**")
                            else:
                                source_link = construct_link(
                                    source, source_path=RAG_SOURCE_PATH, base_url=RAG_BASE_URL
                                )
                                # Display the absolute position in search results
                                result_number = start_idx + idx + 1
                                st.markdown(f"**{result_number}. {source_link}{page_info}**", unsafe_allow_html=True)
                            
                            # Display score if semantic search
                            if search_type == "Semantic" and hasattr(doc, 'metadata') and 'score' in doc.metadata:
                                st.markdown(f"*Similarity: {doc.metadata['score']:.3f}*")
                            
                            # Display highlighted content if available
                            content_to_display = ""
                            
                            if search_type == "Keyword" and 'hl_page_content' in doc.metadata:
                                # Whoosh highlights use B tags - we'll convert them to stylized spans
                                highlighted_text = doc.metadata.get('hl_page_content', '')
                                
                                # Add some CSS to style the highlights properly
                                st.markdown("""
                                <style>
                                .search-highlight {
                                    background-color: #FFFF00;
                                    padding: 0px 2px;
                                    border-radius: 2px;
                                    font-weight: bold;
                                }
                                </style>
                                """, unsafe_allow_html=True)
                                
                                # Convert any B tags to our custom highlighted spans
                                import re
                                content_with_highlights = re.sub(
                                    r'<b class="match term\d*">(.*?)</b>', 
                                    r'<span class="search-highlight">\1</span>', 
                                    highlighted_text
                                )
                                
                                # Show the highlighted content with proper context
                                content_to_display = f"{content_with_highlights}"
                            else:
                                content = doc.page_content
                                # Truncate content if it's too long
                                if len(content) > 300:
                                    content = content[:300] + "..."
                                content_to_display = content
                            
                            # Display the content
                            st.markdown(content_to_display, unsafe_allow_html=True)
                        
                        with col2:
                            # Download button for text content
                            if st.session_state.deduplicate_sources and 'chunk_count' in doc.metadata:
                                # For consolidated documents, show how many chunks were combined
                                chunk_count = doc.metadata.get('chunk_count', 1)
                                st.download_button(
                                    label=f"Download Text ({chunk_count} chunks)",
                                    data=doc.page_content,
                                    file_name=f"document_{result_number}_combined.txt",
                                    mime="text/plain",
                                    key=f"download_text_{result_number}"
                                )
                            else:
                                # Regular download button for single chunk
                                st.download_button(
                                    label="Download Text",
                                    data=doc.page_content,
                                    file_name=f"document_{result_number}.txt",
                                    mime="text/plain",
                                    key=f"download_text_{result_number}"
                                )
                            
                            # For Windows, add an additional download button for the original file
                            if is_windows and os.path.exists(source):
                                file_data, mime_type = get_file_data(source)
                                if file_data:
                                    st.download_button(
                                        label="Download Original",
                                        data=file_data,
                                        file_name=os.path.basename(source),
                                        mime=mime_type,
                                        key=f"download_orig_{result_number}"
                                    )
                        
                        # Create an expander for document details
                        with st.expander("Document Details", expanded=False):
                            # Initialize toggle states in session state if not present
                            if f"show_metadata_{result_number}" not in st.session_state:
                                st.session_state[f"show_metadata_{result_number}"] = False
                            if f"show_full_{result_number}" not in st.session_state:
                                st.session_state[f"show_full_{result_number}"] = False
                                
                            # Add toggle for metadata
                            show_metadata = st.toggle(
                                "Show Document Metadata", 
                                key=f"metadata_toggle_{result_number}",
                                value=st.session_state[f"show_metadata_{result_number}"]
                            )
                            st.session_state[f"show_metadata_{result_number}"] = show_metadata
                            
                            # Add toggle for full content
                            #show_full = st.toggle(
                                #"Show Full Document Content", 
                                #key=f"content_toggle_{result_number}",
                                #value=st.session_state[f"show_full_{result_number}"]
                            #)
                            show_full = 0
                            st.session_state[f"show_full_{result_number}"] = show_full
                            
                            # Show metadata if toggle is enabled
                            if show_metadata:
                                # Create a nice looking metadata display
                                metadata = {k: v for k, v in doc.metadata.items() 
                                         if k not in ['hl_page_content', 'page_content']}
                                
                                # Add a note about consolidated chunks if de-duplication is enabled
                                if st.session_state.deduplicate_sources and 'chunk_count' in doc.metadata:
                                    st.markdown(f"### Document Metadata (Consolidated from {doc.metadata['chunk_count']} chunks)")
                                else:
                                    st.markdown("### Document Metadata")
                                
                                # Create a table view of metadata for better readability
                                metadata_df = []
                                for key, value in metadata.items():
                                    if isinstance(value, str) and not value: continue
                                    metadata_df.append({"Field": key, "Value": str(value)})
                                    
                                if metadata_df:
                                    st.table(metadata_df)
                                else:
                                    st.info("No metadata available for this document.")
                            
                            # Show full content if toggle is enabled
                            if show_full:
                                st.markdown("### Full Document Content")
                                st.markdown(doc.page_content)
                        
                        st.markdown("---")
                
                # Add pagination controls
                if total_pages > 1:
                    st.markdown(f"**Page {st.session_state.current_page} of {total_pages}**")
                    
                    # Create pagination controls
                    pagination_cols = st.columns([1, 1, 3, 1, 1])
                    
                    # First page button
                    with pagination_cols[0]:
                        if st.session_state.current_page > 1:
                            if st.button("❮❮ First", key="first_page", use_container_width=True):
                                st.session_state.current_page = 1
                                st.rerun()
                    
                    # Previous page button
                    with pagination_cols[1]:
                        if st.session_state.current_page > 1:
                            if st.button("❮ Previous", key="prev_page", use_container_width=True):
                                st.session_state.current_page -= 1
                                st.rerun()
                    
                    # Page selector in the middle column
                    with pagination_cols[2]:
                        page_options = list(range(1, total_pages + 1))
                        current_index = min(page_options.index(st.session_state.current_page), len(page_options) - 1)
                        
                        def on_page_change():
                            st.session_state.current_page = st.session_state.page_selector
                            st.rerun()
                            
                        st.selectbox("Go to page:", 
                                    page_options,
                                    index=current_index,
                                    key="page_selector",
                                    on_change=on_page_change)
                    
                    # Next page button
                    with pagination_cols[3]:
                        if st.session_state.current_page < total_pages:
                            if st.button("Next ❯", key="next_page", use_container_width=True):
                                st.session_state.current_page += 1
                                st.rerun()
                    
                    # Last page button
                    with pagination_cols[4]:
                        if st.session_state.current_page < total_pages:
                            if st.button("Last ❯❯", key="last_page", use_container_width=True):
                                st.session_state.current_page = total_pages
                                st.rerun()
                                
            except Exception as e:
                st.error(f"Error processing search: {str(e)}")
                #import traceback
                #st.error(traceback.format_exc())

if __name__ == "__main__":
    # Set page to wide mode when run directly
    st.set_page_config(
        page_title="Search", 
        page_icon="🔍", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        # Get config and check if vectorstore supports keyword search
        cfg = read_config()[0]
        
        try:
            # Try to determine the vectorstore type
            llm = load_llm()
            vectorstore = llm.load_vectorstore()
            has_keyword_search = not isinstance(vectorstore, DenseStore)
        except:
            # If we can't load the vectorstore, default to assuming dense (semantic only)
            has_keyword_search = False
        
        st.session_state.initialized = True
        st.session_state.search_query = ""
        # Set default search type based on what's available
        st.session_state.search_type = "Keyword" if has_keyword_search else "Semantic"
        st.session_state.where_document = ""
        st.session_state.results_limit = 5000 
        st.session_state.deduplicate_sources = True  # Enable deduplication by default
        # Pagination settings
        st.session_state.current_page = 1
        st.session_state.results_per_page = 10
        # We don't need to initialize all the show_full and show_metadata flags here
        # They'll be created dynamically based on the actual search results
    
    main()
