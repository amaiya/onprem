import os
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from langchain.callbacks.base import BaseCallbackHandler
from onprem import LLM, utils as U


def hide_webapp_sidebar_item():
    """
    Hides the webapp.py item from the sidebar navigation
    """
    hide_webapp_style = """
        <style>
            [data-testid="stSidebarNav"] ul li:first-child {
                display: none;
            }
        </style>
    """
    st.markdown(hide_webapp_style, unsafe_allow_html=True)


@st.cache_resource
def load_llm():
    """
    Load the LLM model with caching
    """
    from onprem.app.webapp import read_config
    llm_config = read_config()[0]["llm"]
    return LLM(confirm=False, **llm_config)


@st.cache_resource
def get_embedding_model():
    """
    Load the embedding model with caching
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def compute_similarity(sentence1, sentence2):
    """
    Compute cosine similarity between two sentences
    """
    model = get_embedding_model()
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
    return cosine_score.cpu().numpy()[0][0]


class StreamHandler(BaseCallbackHandler):
    """
    Callback handler for streaming LLM responses
    """
    def __init__(self, container, initial_text="", display_method="markdown"):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token + ""
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")


def setup_llm():
    """
    Set up the LLM with a stream handler
    """
    llm = load_llm()
    chat_box = st.empty()
    stream_handler = StreamHandler(chat_box, display_method="write")
    _ = llm.load_llm()
    llm.llm.callbacks = [stream_handler]
    return llm


def check_create_symlink(source_path, base_url):
    """
    Symlink to folder named <n> in datadir from streamlit's static folder
    
    Supports cross-platform (Windows, Linux, macOS) symlink creation with fallbacks
    """
    if base_url or not source_path:
        return source_path, base_url

    # set new source path
    new_source_path = os.path.dirname(source_path)
    symlink_name = os.path.basename(source_path)

    # check for existence
    staticdir = os.path.join(os.path.dirname(st.__file__), "static")
    symlink_path = os.path.join(staticdir, symlink_name)
    
    # If symlink already exists, use it
    if os.path.islink(symlink_path) or (os.name == 'nt' and os.path.isdir(symlink_path) and os.path.exists(symlink_path)):
        return new_source_path, symlink_name

    # Platform-specific symlink creation
    try:
        if os.name == 'nt':  # Windows
            # Windows requires administrator privileges for symlinks unless developer mode is enabled
            # First try with the appropriate flags for Windows
            try:
                # Use directory junction which doesn't require admin rights
                import subprocess
                target_path = os.path.abspath(new_source_path)
                subprocess.check_call(['mklink', '/J', symlink_path, target_path], shell=True)
            except Exception as e:
                # Try with standard os.symlink with required Windows flags
                # 1 = SYMBOLIC_LINK_FLAG_DIRECTORY, needed for directory symlinks
                # 0x02 = SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE, for non-admin creation if developer mode is on
                os.symlink(new_source_path, symlink_path, target_is_directory=True)
        else:
            # Linux/macOS symlink creation
            os.symlink(new_source_path, symlink_path)
            
        return new_source_path, symlink_name
    except Exception as e:
        # If symlink creation fails, print a helpful message but continue
        st.warning(f"""
        Unable to create symlink for document viewing: {str(e)}
        
        To fix this on Windows:
        1. Enable Developer Mode in Windows Settings > For Developers, OR
        2. Run the application as administrator, OR
        3. Open Documents directly from the file system
        
        Continuing without document preview capability.
        """)
        return source_path, base_url


def construct_link(filepath, source_path=None, base_url=None):
    """
    Constructs a link to a document
    
    For Windows compatibility:
    - If symlinks/directory junctions work, it will use Streamlit's static file server
    - If symlinks fail, it can generate a local file:// URL as a fallback (for direct file system access)
    - Or you can provide a base_url to use an external web server for serving files
    """
    import urllib
    from pathlib import Path

    filename = os.path.basename(filepath)
    if source_path is None:
        return filename
    
    # For Windows compatibility - check if we should use file:// URLs as fallback
    # This is used when symlinks can't be created and files need to be accessed directly
    use_file_url = False
    if os.name == 'nt' and not base_url:
        # Check if the base URL is a directory on the filesystem
        # rather than a symlink in Streamlit's static folder
        staticdir = os.path.join(os.path.dirname(st.__file__), "static")
        symlink_name = os.path.basename(source_path)
        symlink_path = os.path.join(staticdir, symlink_name)
        if not (os.path.islink(symlink_path) or os.path.isdir(symlink_path)):
            use_file_url = True
    
    if use_file_url:
        # Use file:// URL for direct file system access (Windows fallback)
        abs_path = os.path.abspath(filepath)
        file_url = f"file:///{abs_path.replace(os.sep, '/')}"
        return (
            f'<a href="{file_url}" '
            + f'target="_blank" title="Click to view original source (local file)">{filename}</a>'
        )
    else:
        # Standard web URL (using Streamlit static server or custom base_url)
        base_url = base_url or "/"
        relative = str(Path(filepath).relative_to(source_path))
        link = os.path.join(base_url, relative)
        return (
            f'<a href="{urllib.parse.quote(link)}" '
            + f'target="_blank" title="Click to view original source">{filename}</a>'
        )




from pyparsing import (
    Word, QuotedString, Suppress, Group, Forward, 
    ZeroOrMore, alphanums, alphas, nums, Optional as PPOptional, 
    Literal, CaselessLiteral, infixNotation, opAssoc, ParserElement
)
from typing import Dict, List, Union, Any, Optional

def lucene_to_chroma(query_str: str) -> Dict[str, Union[Dict, List, None]]:
    """
    Transform a Lucene-style query into Chroma query parameters using pyparsing.
    
    Args:
        query_str: A Lucene-style query string (e.g., '"climate change" AND extension:(pdf OR docx)')
        
    Returns:
        Dictionary with 'where_document' and 'filter' parameters for Chroma
    """
    # Enable more helpful error messages
    ParserElement.setDefaultWhitespaceChars(" \t")
    
    # Define the parser elements
    AND = CaselessLiteral("AND")
    OR = CaselessLiteral("OR")
    NOT = CaselessLiteral("NOT")
    
    # Define basic term types
    term = Word(alphanums + "_-.'@")
    field_name = Word(alphas, alphanums + "_")
    
    # Define quoted string
    quoted_string = QuotedString('"', escChar='\\', unquoteResults=True)
    
    # Forward declaration for expressions inside field value parentheses
    field_value_expr = Forward()
    
    # Define simple value and complex values (for things like extension:(pdf OR docx))
    simple_value = quoted_string | Word(alphanums + ".-_")
    complex_value = Suppress("(") + field_value_expr + Suppress(")")
    
    # Define a single value or multiple values with operators for field values
    field_value_term = simple_value
    
    # Define the expression with boolean operators for field values
    field_value_expr << infixNotation(
        field_value_term,
        [
            (CaselessLiteral("NOT"), 1, opAssoc.RIGHT),
            (CaselessLiteral("AND"), 2, opAssoc.LEFT),
            (CaselessLiteral("OR"), 2, opAssoc.LEFT),
        ]
    )
    
    # Define field:value expression for metadata filters
    field_value = Group(field_name + Suppress(":") + (complex_value | simple_value))
    
    # Define various expressions that can appear in the query
    expression = Forward()
    
    # Define a content term (either a quoted string or a simple term)
    content_term = quoted_string | term
    
    # Define a factor (either content term, field:value, or parenthesized expression)
    factor = field_value | content_term | (Suppress("(") + expression + Suppress(")"))
    
    # Define the expression with boolean operators
    expression << infixNotation(
        factor,
        [
            (NOT, 1, opAssoc.RIGHT),
            (AND, 2, opAssoc.LEFT),
            (OR, 2, opAssoc.LEFT),
        ]
    )
    
    # Parse the query string
    try:
        if not query_str.strip():
            return {"where_document": {}, "filter": None}
        
        parsed_result = expression.parseString(query_str, parseAll=True)
        
        # Process the parsed result to extract content terms and metadata filters
        content_terms = []
        metadata_filters = []
        
        def process_value(value):
            """Convert string values to appropriate types"""
            if isinstance(value, str):
                if value.lower() == "true":
                    return True
                elif value.lower() == "false":
                    return False
                elif value.isdigit():
                    return int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                    return float(value)
            return value
        
        def process_field_value_expr(result, field_name):
            """Process a complex field value expression like (pdf OR docx)"""
            if not isinstance(result, list):
                # Single value
                processed_value = process_value(result)
                return {field_name: {"$eq": processed_value}}
            
            # Check if this is an OR expression
            if len(result) >= 3 and result[1] == "OR":
                or_conditions = []
                for i in range(0, len(result), 2):
                    if isinstance(result[i], list):
                        or_conditions.append(process_field_value_expr(result[i], field_name))
                    else:
                        processed_value = process_value(result[i])
                        or_conditions.append({field_name: {"$eq": processed_value}})
                return {"$or": or_conditions}
            
            # Check if this is an AND expression
            elif len(result) >= 3 and result[1] == "AND":
                and_conditions = []
                for i in range(0, len(result), 2):
                    if isinstance(result[i], list):
                        and_conditions.append(process_field_value_expr(result[i], field_name))
                    else:
                        processed_value = process_value(result[i])
                        and_conditions.append({field_name: {"$eq": processed_value}})
                return {"$and": and_conditions}
            
            # For NOT expressions or other complex cases
            elif result[0] == "NOT" and len(result) > 1:
                if isinstance(result[1], list):
                    # Process nested expressions
                    return {"$not": process_field_value_expr(result[1], field_name)}
                else:
                    processed_value = process_value(result[1])
                    return {field_name: {"$ne": processed_value}}
            
            # Fallback for other structures
            else:
                # Just return the first value if we can't parse the structure
                if isinstance(result[0], list):
                    return process_field_value_expr(result[0], field_name)
                processed_value = process_value(result[0])
                return {field_name: {"$eq": processed_value}}
            
        def process_parsed_result(result):
            nonlocal content_terms, metadata_filters
            
            if isinstance(result, list) or isinstance(result, tuple):
                # Check if it's a field:value pair (metadata filter)
                if len(result) == 2 and isinstance(result[0], str):
                    field = result[0]
                    value = result[1]
                    
                    # Handle complex expressions like extension:(pdf OR docx)
                    if isinstance(value, list):
                        filter_expr = process_field_value_expr(value, field)
                        metadata_filters.append(filter_expr)
                    else:
                        # Simple field:value case
                        processed_value = process_value(value)
                        metadata_filters.append({field: {"$eq": processed_value}})
                    return
                
                # Process nested structures
                for item in result:
                    if item in ('AND', 'OR', 'NOT'):
                        continue
                    process_parsed_result(item)
            elif isinstance(result, str) and result not in ('AND', 'OR', 'NOT'):
                # It's a content term
                content_terms.append(result)
        
        process_parsed_result(parsed_result.asList())
        
        # Build the where_document parameter for content search
        where_document = {}
        if content_terms:
            where_document = {"$contains": " ".join(content_terms)}
        
        # Build the filter parameter for metadata
        filter_param = None
        if metadata_filters:
            if len(metadata_filters) == 1:
                filter_param = metadata_filters[0]
            else:
                filter_param = {"$and": metadata_filters}
        
        return {
            "where_document": where_document,
            "filter": filter_param
        }
    
    except Exception as e:
        # Return empty query parameters on parsing error
        return {"where_document": {}, "filter": None, "error": str(e)}
