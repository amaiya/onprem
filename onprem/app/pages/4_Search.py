import os
import sys
import streamlit as st
from typing import Type

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
    RAG_SOURCE_PATH, RAG_BASE_URL = check_create_symlink(RAG_SOURCE_PATH, RAG_BASE_URL)
    
    # Setup header
    st.header("Document Search")
    st.markdown("""
    Search through your indexed documents using keywords and filters.
    """)
    
    # Check if vector database exists
    if not VECTORDB_PATH or not os.path.exists(VECTORDB_PATH):
        st.error("No vector database found. Please ingest documents first.")
        return
    
    # Initialize the vector store using LLM.load_vectorstore()
    try:
        # Load the LLM instance from the configuration
        llm = load_llm()
        
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
        st.session_state.results_limit = 20
    
    # Create search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("Enter search terms:", 
                              placeholder="Enter keywords to search for",
                              value=st.session_state.search_query)
    
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
        results_limit = st.slider("Number of results to display:", 
                                min_value=5, 
                                max_value=100, 
                                value=st.session_state.results_limit, 
                                step=5)
    
    # Search and reset buttons in columns
    col1, col2 = st.columns([4, 1])
    with col1:
        search_button = st.button("Search", type="primary")
    with col2:
        reset_button = st.button("Reset", type="secondary")
    
    # Reset search state if reset button is clicked
    if reset_button:
        st.session_state.search_query = ""
        # Set default search type based on available search types
        st.session_state.search_type = "Keyword" if has_keyword_search else "Semantic"
        st.session_state.where_document = ""
        st.session_state.results_limit = 20
        st.rerun()
        
    # Update session state when search button is clicked
    if search_button and query:
        st.session_state.search_query = query
        st.session_state.search_type = search_type
        st.session_state.where_document = where_document
        st.session_state.results_limit = results_limit

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
                
                # Display results
                st.subheader(f"Search Results: {total_hits} documents found")
                
                if not hits:
                    st.info("No documents found matching your search criteria.")
                    return
                
                # Display each result
                for i, doc in enumerate(hits):
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        
                        source = doc.metadata.get("source", "Unknown source")
                        source_link = construct_link(
                            source, source_path=RAG_SOURCE_PATH, base_url=RAG_BASE_URL
                        )
                        
                        # Get page if available - only show for PDFs
                        page = doc.metadata.get("page")
                        extension = doc.metadata.get("extension", "").lower()
                        page_info = ""
                        if page is not None and extension == "pdf":
                            page_info = f", page {page + 1}"
                        
                        with col1:
                            st.markdown(f"**{i+1}. {source_link}{page_info}**", unsafe_allow_html=True)
                            
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
                            st.download_button(
                                label=f"Download",
                                data=doc.page_content,
                                file_name=f"document_{i+1}.txt",
                                mime="text/plain",
                                key=f"download_{i}"
                            )
                        
                        # Show full content if requested
                        if st.session_state.get(f"show_full_{i}", False):
                            with st.expander("Full Content", expanded=True):
                                st.markdown(doc.page_content)
                                
                                # Show additional metadata
                                with st.expander("Document Metadata"):
                                    metadata = {k: v for k, v in doc.metadata.items() 
                                              if k not in ['hl_page_content', 'page_content']}
                                    st.json(metadata)
                        
                        st.markdown("---")
                        
            except Exception as e:
                st.error(f"Error processing search: {str(e)}")
                #import traceback
                #st.error(traceback.format_exc())

if __name__ == "__main__":
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
        st.session_state.results_limit = 20
        for i in range(100):  # Reasonable limit for number of results
            st.session_state[f"show_full_{i}"] = False
    
    main()
