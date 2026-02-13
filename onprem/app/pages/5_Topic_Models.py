import os
import sys
import streamlit as st
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
from collections import Counter

# Add parent directory to path to allow importing when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from parent modules
from OnPrem import read_config
from utils import setup_llm, load_llm, construct_link, check_create_symlink
from utils import lucene_to_chroma, get_prompt_template
from onprem.app.utils import load_vectorstore
from onprem.ingest.stores.sparse import SparseStore
from onprem.ingest.stores.dense import DenseStore
from onprem.ingest.stores.dual import DualStore
from onprem.sk.tm import TopicModel


def resolve_path(path):
    """
    Resolves any format strings in the given path
    
    Args:
        path: The path that might contain format strings
        
    Returns:
        The resolved path
    """
    if not path:
        return path
        
    # Import utils
    from onprem import utils as U
    
    # Handle common format strings
    resolved = path
    if "{webapp_dir}" in path:
        resolved = resolved.replace("{webapp_dir}", U.get_webapp_dir())
    if "{models_dir}" in path:
        resolved = resolved.replace("{models_dir}", U.get_models_dir())
    if "{datadir}" in path:
        resolved = resolved.replace("{datadir}", U.get_datadir())
        
    return resolved


def generate_wordcloud(word_dict, max_words=50):
    """
    Generate a word cloud from a dictionary of word frequencies
    
    Args:
        word_dict: Dictionary with words as keys and frequencies as values
        max_words: Maximum number of words to display
    """
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            relative_scaling=0.5,
            colormap='viridis'
        ).generate_from_frequencies(word_dict)
        
        # Display using matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        plt.close()
    except ImportError:
        st.info("💡 Install wordcloud package to see visual word clouds: `pip install wordcloud`")
        # Fallback: display as text
        st.write("**Top Keywords:**")
        sorted_words = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        cols = st.columns(4)
        for i, (word, count) in enumerate(sorted_words):
            cols[i % 4].write(f"• {word} ({count})")


def prepare_excel_file(results_df):
    """Create a formatted Excel file in memory"""
    # Create Excel file in memory with enhanced formatting
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        results_df.to_excel(writer, index=False)
        
        # Apply formatting
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        
        from openpyxl.styles import Font, Alignment, PatternFill
        bold_font = Font(bold=True)
        header_fill = PatternFill(start_color="E0E0E0", end_color="E0E0E0", fill_type="solid")
        
        # Apply formatting to headers
        for col_num, column_title in enumerate(results_df.columns):
            cell = worksheet.cell(row=1, column=col_num+1)
            cell.font = bold_font
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.fill = header_fill
        
        # Auto-adjust columns' width
        for column_cells in worksheet.columns:
            length = max(len(str(cell.value) or "") for cell in column_cells)
            adjusted_width = min(length + 8, 100)
            worksheet.column_dimensions[column_cells[0].column_letter].width = adjusted_width
    
    output.seek(0)
    return output


def display_preview(content, max_length=200):
    """Display a preview of content with truncation"""
    if len(content) > max_length:
        return content[:max_length] + "..."
    return content


def main():
    """
    Page for topic modeling on ingested documents
    """
    # Handle reset before any widgets are rendered
    if "reset_topics" in st.session_state and st.session_state.reset_topics:
        # Remove topic-related session state
        keys_to_remove = [
            'topic_model', 'topic_df', 'topic_dict', 'topic_docs',
            'selected_topic', 'topic_page', 'exclude_terms', 'reset_topics'
        ]
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    cfg = read_config()[0]
    
    # Get configuration
    RAG_SOURCE_PATH = cfg.get("ui", {}).get("rag_source_path", None)
    RAG_BASE_URL = cfg.get("ui", {}).get("rag_base_url", None)
    VECTORDB_PATH = cfg.get("llm", {}).get("vectordb_path", None)
    STORE_TYPE = cfg.get("llm", {}).get("store_type", "dense")
    
    # Resolve paths
    from onprem import utils as U
    
    DOCUMENTS_PATH = RAG_SOURCE_PATH
    if RAG_SOURCE_PATH and "{webapp_dir}" in RAG_SOURCE_PATH:
        DOCUMENTS_PATH = RAG_SOURCE_PATH.replace("{webapp_dir}", U.get_webapp_dir())
    
    # Load LLM to get model name
    llm = load_llm()
    MODEL_NAME = llm.model_name
    
    RAG_SOURCE_PATH, RAG_BASE_URL = check_create_symlink(RAG_SOURCE_PATH, RAG_BASE_URL)
    
    # Add some CSS for better styling
    st.markdown("""
    <style>
    .stChatMessage {
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # Page header
    st.markdown("""
    <h1 style="
        color: #0068c9;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0068c9;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
    ">
        <span style="font-size: 1.8rem; margin-right: 0.5rem;">🎯</span> 
        Topic Discovery
    </h1>
    """, unsafe_allow_html=True)
    
    # Info card
    st.markdown("""
    <div style="
        padding: 12px 15px;
        border-radius: 8px;
        margin-bottom: 25px;
        background: linear-gradient(to right, rgba(0, 104, 201, 0.05), rgba(92, 137, 229, 0.1));
        border-left: 4px solid #0068c9;
        display: flex;
        align-items: center;
    ">
        <span style="font-size: 1.2rem; margin-right: 10px;">🎯</span>
        <div>
            <p style="margin: 0; font-weight: 500;">Discover hidden themes in your document collection</p>
            <p style="margin: 3px 0 0 0; font-size: 0.85rem; color: #666;">
                This analysis uses topic modeling to identify subjects and themes buried in search results.
            </p>
        </div>
    </div>
    """  , unsafe_allow_html=True)
    
    st.markdown("""
    > **Theme Discovery** will discover subjects buried in the current search results and identify documents most relevant to each subject.
    """)
    
    st.markdown("**Note:** This analysis employs a method that sacrifices accuracy for speed.")
    st.markdown("---")
    
    # Check if vector database exists
    if not VECTORDB_PATH or not os.path.exists(VECTORDB_PATH):
        st.error("No vector database found. Please ingest documents first by going to Manage.")
        return
    
    # Initialize the vector store
    try:
        vectorstore = load_vectorstore()
        store_type = llm.get_store_type()
        
        if not vectorstore.exists():
            st.error("No documents have been indexed. Please ingest documents first by going to Manage.")
            return
            
        # Determine if keyword search is available
        has_keyword_search = not isinstance(vectorstore, DenseStore)
            
    except Exception as e:
        st.error(f"Error loading search index: {str(e)}")
        return
    
    # Initialize session state
    if 'topic_page' not in st.session_state:
        st.session_state.topic_page = 1
    if 'exclude_terms' not in st.session_state:
        st.session_state.exclude_terms = []
    
    # Get folders from DOCUMENTS_PATH
    folder_options = ["All folders"]
    if DOCUMENTS_PATH and os.path.exists(DOCUMENTS_PATH):
        subfolders = [d for d in os.listdir(DOCUMENTS_PATH) 
                     if os.path.isdir(os.path.join(DOCUMENTS_PATH, d))]
        folder_options.extend(subfolders)
    
    # Show results if already computed
    if 'topic_df' in st.session_state and 'topic_dict' in st.session_state:
        st.subheader("Results")
        
        theme_dict = st.session_state.topic_dict
        df = st.session_state.topic_df
        
        # Create topic selection menu
        topic_ids = [k for k in theme_dict]
        topic_desc = [f"{theme_dict[k][0]}" + f" ({theme_dict[k][1]} docs)" for k in theme_dict]
        topic_ids = [-1] + topic_ids
        topic_desc = ["-Select a Topic-"] + topic_desc
        topic_dict_display = dict([(topic_id, topic_desc[i]) for i, topic_id in enumerate(topic_ids)])
        
        st.write(
            "The menu below shows the discovered themes represented by lists of representative words. "
            + "Make a selection to view documents most representative of the theme."
        )
        
        if len(st.session_state.exclude_terms) > 0:
            st.write(f"Note: The following terms have been excluded from results: *{', '.join(st.session_state.exclude_terms)}*")
        
        topic_id = st.selectbox(
            "Select a discovered theme", 
            topic_ids, 
            format_func=lambda x: f"{topic_dict_display[x]}"
        )
        
        # Export options
        st.markdown("")
        col1, col2, col3 = st.columns([1.2, 1.2, 2])
        
        # Export theme summary
        theme_summary_df = pd.DataFrame.from_dict(theme_dict, orient="index")
        theme_summary_df = theme_summary_df.reset_index()
        theme_summary_df.columns = ["Theme ID", "Theme", "Number of Docs"]
        summary_excel = prepare_excel_file(theme_summary_df)
        
        col1.download_button(
            label="Export Theme Summary",
            help="Export the table with all themes and number of documents for each.",
            data=summary_excel,
            file_name=f"theme_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Show topic-specific results if a topic is selected
        if topic_id != -1:
            # Filter dataframe for selected topic
            topic_df = df.loc[df['topic_id'] == topic_id].sort_values('topic_score', ascending=False)
            
            # Export selected topic results
            topic_excel = prepare_excel_file(topic_df)
            col2.download_button(
                label="Export Selected Topic",
                help="Export the documents that match the selected topic ID.",
                data=topic_excel,
                file_name=f"theme_topic_{topic_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.markdown("")
            
            # Generate word cloud from top keywords
            if 'top_keywords' in topic_df.columns:
                top_keywords = []
                for lst in list(topic_df['top_keywords'].values):
                    if lst:
                        top_keywords.extend(lst)
                
                if top_keywords:
                    cnt = Counter(top_keywords)
                    wc_d = dict(cnt.most_common(50))
                    generate_wordcloud(wc_d)
            
            # Pagination
            st.markdown("<div id='linkto_top'></div>", unsafe_allow_html=True)
            total_hits = topic_df.shape[0]
            page_size = 20
            total_pages = max(1, (total_hits + page_size - 1) // page_size)
            
            col1, col2 = st.columns((1, 5))
            with col1:
                value = int(st.session_state.topic_page) if 'topic_page' in st.session_state else 1
                if value > total_pages:
                    value = 1
                page = st.number_input(
                    "Page Number",
                    min_value=1,
                    max_value=total_pages,
                    value=value,
                    key="topic_page",
                    label_visibility="hidden"
                )
            with col2:
                st.markdown("")
                st.markdown(f"of {total_pages} pages ({total_hits} results)")
            
            # Display results table
            st.markdown("---")
            colms = st.columns((1, 3, 2, 3))
            fields = ["**No. (Score)**", "**Source**", "**Preview**", "**Top Keywords**"]
            for col, field_name in zip(colms, fields):
                col.markdown(field_name)
            
            st.markdown("---")
            
            # Show paginated results
            offset = int((page - 1) * page_size)
            df_page = topic_df[int(offset):int(offset + page_size)]
            
            for j, (i, row) in enumerate(df_page.iterrows(), 1):
                col1, col2, col3, col4 = st.columns((1, 3, 2, 3))
                
                # Number and score
                col1.write(f"{offset + j}: ({row['topic_score']:.2f})")
                
                # Source
                source = row.get('source', 'Unknown')
                if DOCUMENTS_PATH and source.startswith(DOCUMENTS_PATH):
                    source = source[len(DOCUMENTS_PATH):].lstrip('/')
                col2.write(display_preview(source, 100))
                
                # Preview
                content = row.get('content', '')
                col3.write(display_preview(content, 100))
                
                # Top keywords
                top_keywords = row.get('top_keywords', [])
                if top_keywords:
                    keywords_str = " | ".join(top_keywords[:5])
                    col4.write(keywords_str)
                
                st.markdown("---")
            
            st.markdown("<a href='#linkto_top'>Back to Top</a>", unsafe_allow_html=True)
            
            # Show query for documents
            if 'source' in topic_df.columns:
                checked = st.checkbox(
                    f"Show source files for these {topic_df.shape[0]} documents",
                    value=False,
                    help="Shows the list of source files for documents in this topic"
                )
                if checked:
                    unique_sources = topic_df['source'].unique()
                    st.write(f"**{len(unique_sources)} unique source files:**")
                    for src in unique_sources[:20]:  # Show first 20
                        if DOCUMENTS_PATH and src.startswith(DOCUMENTS_PATH):
                            src = src[len(DOCUMENTS_PATH):].lstrip('/')
                        st.write(f"- {src}")
                    if len(unique_sources) > 20:
                        st.write(f"... and {len(unique_sources) - 20} more")
        
        # Add button to start new analysis
        st.markdown("---")
        if st.button("Start New Analysis", type="secondary"):
            st.session_state.reset_topics = True
            st.rerun()
        
        return
    
    # Document Selection Section
    st.subheader("Document Selection")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search terms (optional):", 
            placeholder="Enter keywords to filter documents",
            key="query_input",
            help="Leave empty to analyze all documents in selected folder"
        )
    
    with col2:
        search_options = ["Semantic"]
        if has_keyword_search:
            search_options.insert(0, "Keyword")
        
        search_type = st.selectbox(
            "Search type:", 
            search_options,
            key="search_type"
        )
    
    selected_folder = st.selectbox(
        "Folder to analyze:", 
        folder_options,
        index=0,
        key="folder_selector",
        help="Select a specific folder or 'All folders' to analyze entire collection"
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            exclude_terms_input = st.text_input(
                "Exclude terms (comma-separated):",
                help="Single-word terms to exclude from theme discovery",
                placeholder="e.g., said, also, however"
            )
            
            if exclude_terms_input:
                exclude_terms = [term.strip().lower() for term in exclude_terms_input.split(",") if term.strip()]
                bad_terms = [term for term in exclude_terms if len(term.split()) > 1]
                if bad_terms:
                    st.error(f"Exclude terms must be single words: {bad_terms}")
            else:
                exclude_terms = []
            
            n_topics = st.number_input(
                "Number of topics:",
                min_value=2,
                max_value=500,
                value=None,
                step=1,
                help="Leave empty for automatic determination"
            )
        
        with col2:
            max_results = st.slider(
                "Maximum documents to analyze:",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="More documents = slower but potentially better results"
            )
            
            max_num_words = st.number_input(
                "Max words per document:",
                min_value=500,
                max_value=10000,
                value=5000,
                step=500,
                help="Limit text length for faster processing"
            )
    
    # Check minimum documents
    if selected_folder == "All folders" and not query:
        st.info("💡 **Tip:** Enter search terms or select a specific folder to discover themes.")
    
    st.markdown("---")
    
    # Action buttons
    button_cols = st.columns([1, 1, 4])
    with button_cols[0]:
        discover_button = st.button("Discover Themes", type="primary")
    with button_cols[1]:
        reset_button = st.button("Reset", type="secondary")
    
    if reset_button:
        st.session_state.reset_topics = True
        st.rerun()
    
    if discover_button:
        # Validate inputs
        if selected_folder == "All folders" and not query:
            st.error("Please enter search terms or select a specific folder.")
            st.stop()
        
        try:
            with st.spinner("Retrieving documents..."):
                # Build filters
                folder_filter = None
                where_clause = None
                
                if selected_folder != "All folders":
                    folder_name = selected_folder
                    folder_path = os.path.join(DOCUMENTS_PATH, folder_name)
                    norm_folder_path = os.path.normpath(folder_path).replace('\\', '/')
                    folder_filter = f'source:{norm_folder_path}*'
                    where_clause = folder_filter
                
                query_text = query if query else "*"
                
                # Retrieve documents
                if search_type == "Keyword":
                    results = vectorstore.query(
                        query=query_text,
                        limit=max_results,
                        where_document=where_clause,
                        return_dict=False
                    )
                    hits = results.get('hits', [])
                else:  # Semantic search
                    if store_type != 'sparse':
                        chroma_filters = lucene_to_chroma(where_clause) if where_clause else {}
                        where_document = chroma_filters.get('where_document')
                        filter_options = chroma_filters.get('filter')
                    else:
                        where_document = where_clause
                        filter_options = None
                    
                    if not query:
                        query_text = "document"
                    
                    hits = vectorstore.semantic_search(
                        query=query_text,
                        k=max_results,
                        filters=filter_options,
                        where_document=where_document
                    )
                    
                    # Additional folder filtering for semantic search
                    if selected_folder != "All folders" and query:
                        folder_path = os.path.join(DOCUMENTS_PATH, selected_folder)
                        norm_folder_path = os.path.normpath(folder_path).replace('\\', '/')
                        hits = [hit for hit in hits if 'source' in hit.metadata and 
                                hit.metadata['source'].startswith(norm_folder_path)]
                
                total_hits = len(hits)
                
                if total_hits < 10:
                    st.warning(f"Only found {total_hits} documents. Theme discovery works best with at least 100 documents.")
                    st.stop()
                
                if total_hits < 500:
                    st.info(f"Found {total_hits} documents. Theme discovery typically works better with at least 1000 documents.")
                
                st.success(f"Retrieved {total_hits} documents. Starting topic modeling...")
            
            with st.spinner("Building topic model (this may take a few minutes)..."):
                # Prepare documents
                documents = []
                sources = []
                
                for hit in hits:
                    content = hit.page_content
                    if max_num_words:
                        words = content.split()[:max_num_words]
                        content = " ".join(words)
                    documents.append(content)
                    sources.append(hit.metadata.get('source', 'Unknown'))
                
                # Prepare stop words
                from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
                stop_words = list(ENGLISH_STOP_WORDS) + exclude_terms
                
                # Build topic model
                tm = TopicModel(
                    texts=documents,
                    n_topics=n_topics,
                    min_df=5,
                    max_df=0.5,
                    model_type='nmf',
                    n_features=100000,
                    max_iter=10,
                    stop_words=stop_words,
                    verbose=True
                )
                
                # Build document-topic distribution
                tm.build(documents, threshold=0.05)
                
                # Get topics
                topic_dict = tm.get_topics(show_counts=True)
                
                # Create results dataframe
                df = pd.DataFrame({
                    'content': documents,
                    'source': sources
                })
                
                # Filter based on threshold
                df = tm.filter(df)
                
                # Add topic information
                df['topic_id'] = list(map(np.argmax, tm.doc_topics))
                df['topic_score'] = list(map(np.amax, tm.doc_topics))
                
                # Get top keywords for each document
                top_keywords_list = []
                for topic_id in df['topic_id']:
                    word_weights = tm.get_word_weights(topic_id, n_words=10)
                    keywords = [word for word, weight in word_weights]
                    top_keywords_list.append(keywords)
                
                df['top_keywords'] = top_keywords_list
                
                # Truncate content for display
                df['content'] = [text[:1024] for text in df['content'].values]
                
                # Add topic labels
                df['topic_label'] = df.apply(lambda x: topic_dict[x.topic_id][0], axis=1)
                
                # Store in session state
                st.session_state.topic_df = df
                st.session_state.topic_dict = topic_dict
                st.session_state.exclude_terms = exclude_terms
                st.session_state.topic_page = 1
                
                st.success(f"✅ Topic modeling complete! Discovered {len(topic_dict)} themes.")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error during topic modeling: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    st.set_page_config(
        page_title="Topic Discovery", 
        page_icon="🎯", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    main()
