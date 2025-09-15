# OnPrem Workflow Tutorial

A comprehensive guide to using the OnPrem workflow engine for automated document processing pipelines.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Workflow Structure](#workflow-structure)
4. [Node Types Reference](#node-types-reference)
5. [Configuration Options](#configuration-options)
6. [Advanced Examples](#advanced-examples)
7. [Validation and Error Handling](#validation-and-error-handling)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

The OnPrem workflow engine allows you to define automated document processing pipelines using YAML configuration files. Instead of writing code, you visually design workflows by connecting different types of nodes:

- **Loader Nodes** - Load documents from various sources
- **TextSplitter Nodes** - Process and chunk documents  
- **Storage Nodes** - Store processed documents in vector databases or search indices

The engine validates connections, ensures proper data flow, and executes nodes in the correct order.

## Quick Start

### 1. Create a Simple Workflow

Create a file called `my_workflow.yaml`:

```yaml
nodes:
  document_loader:
    type: LoadFromFolder
    config:
      source_directory: "/path/to/your/documents"
      verbose: true
  
  text_chunker:
    type: SplitByCharacterCount
    config:
      chunk_size: 500
      chunk_overlap: 50
  
  search_index:
    type: WhooshStore
    config:
      persist_location: "my_search_index"

connections:
  - from: document_loader
    from_port: documents
    to: text_chunker
    to_port: documents
  
  - from: text_chunker
    from_port: documents
    to: search_index
    to_port: documents
```

### 2. Run the Workflow

```python
from onprem.pipelines.workflow import execute_workflow

# Execute the workflow
results = execute_workflow("my_workflow.yaml", verbose=True)
```

Or programmatically:

```python
from onprem.pipelines.workflow import WorkflowEngine

engine = WorkflowEngine()
engine.load_workflow_from_yaml("my_workflow.yaml")
results = engine.execute(verbose=True)
```

## Workflow Structure

A workflow YAML file has two main sections:

### Nodes Section

Defines the processing nodes in your pipeline:

```yaml
nodes:
  node_id:              # Unique identifier for this node
    type: NodeTypeName  # Type of node (see Node Types Reference)
    config:             # Configuration specific to this node type
      parameter1: value1
      parameter2: value2
```

### Connections Section

Defines how data flows between nodes:

```yaml
connections:
  - from: source_node_id    # Source node ID
    from_port: output_port  # Output port name
    to: target_node_id      # Target node ID  
    to_port: input_port     # Input port name
```

## Node Types Reference

### Loader Nodes

Loader nodes read documents from various sources and output `List[Document]`.

#### LoadFromFolder

Loads all documents from a directory using `ingest.load_documents`.

```yaml
nodes:
  my_loader:
    type: LoadFromFolder
    config:
      source_directory: "/path/to/documents"    # Required: Directory path
      ignored_files: ["temp.txt", "draft.doc"] # Optional: Files to skip
      verbose: true                             # Optional: Show progress
      pdf_markdown: false                       # Optional: Convert PDFs to markdown
      pdf_unstructured: false                   # Optional: Use unstructured PDF parsing
      n_proc: null                             # Optional: Number of CPU cores (null = all)
      store_md5: false                         # Optional: Store MD5 hash in metadata
      store_mimetype: false                    # Optional: Store MIME type in metadata
      store_file_dates: false                  # Optional: Store file dates in metadata
      infer_table_structure: false             # Optional: Extract tables from PDFs
      caption_tables: false                    # Optional: Generate table captions (requires llm)
      extract_document_titles: false           # Optional: Extract document titles (requires llm)
```

**Output Ports:**
- `documents`: `List[Document]` - Loaded documents

#### LoadSingleDocument

Loads a single document using `ingest.load_single_document`.

```yaml
nodes:
  single_doc:
    type: LoadSingleDocument
    config:
      file_path: "/path/to/document.pdf"       # Required: Path to single file
      pdf_markdown: false                      # Optional: Convert PDF to markdown
      pdf_unstructured: false                  # Optional: Use unstructured parsing
      store_md5: false                        # Optional: Store MD5 hash
      store_mimetype: false                   # Optional: Store MIME type
      store_file_dates: false                 # Optional: Store file dates
      infer_table_structure: false            # Optional: Extract tables
```

**Output Ports:**
- `documents`: `List[Document]` - Loaded document

#### LoadWebDocument

Downloads and loads a document from a URL.

```yaml
nodes:
  web_doc:
    type: LoadWebDocument
    config:
      url: "https://example.com/document.pdf"  # Required: Document URL
      username: "user"                         # Optional: Authentication username
      password: "pass"                         # Optional: Authentication password
```

**Output Ports:**
- `documents`: `List[Document]` - Downloaded document

### TextSplitter Nodes

TextSplitter nodes process documents and output chunked `List[Document]`.

#### SplitByCharacterCount

Chunks documents by character count using `ingest.chunk_documents`.

```yaml
nodes:
  char_splitter:
    type: SplitByCharacterCount
    config:
      chunk_size: 500                         # Optional: Characters per chunk (default: 500)
      chunk_overlap: 50                       # Optional: Overlap between chunks (default: 50)
      infer_table_structure: false            # Optional: Handle tables specially
      preserve_paragraphs: false              # Optional: Keep paragraphs intact
```

**Input Ports:**
- `documents`: `List[Document]` - Documents to chunk

**Output Ports:**
- `documents`: `List[Document]` - Chunked documents

#### SplitByParagraph

Chunks documents by paragraph boundaries, preserving document structure.

```yaml
nodes:
  para_splitter:
    type: SplitByParagraph
    config:
      chunk_size: 1000                        # Optional: Max characters per chunk
      chunk_overlap: 100                      # Optional: Overlap between chunks
      # preserve_paragraphs is automatically set to True
```

**Input Ports:**
- `documents`: `List[Document]` - Documents to chunk

**Output Ports:**
- `documents`: `List[Document]` - Paragraph-based chunks

#### KeepFullDocument

Passes documents through without any chunking (useful for small documents).

```yaml
nodes:
  no_split:
    type: KeepFullDocument
    config: {}  # No configuration needed
```

**Input Ports:**
- `documents`: `List[Document]` - Documents to pass through

**Output Ports:**
- `documents`: `List[Document]` - Unchanged documents

### Storage Nodes

Storage nodes save documents to various backends and return status messages.

#### ChromaStore

Stores documents in a ChromaDB vector database.

```yaml
nodes:
  vector_db:
    type: ChromaStore
    config:
      persist_location: "/path/to/chromadb"    # Optional: Database path
      # Additional ChromaDB configuration options
```

**Input Ports:**
- `documents`: `List[Document]` - Documents to store

**Output Ports:**
- `status`: `str` - Storage status message

#### WhooshStore

Stores documents in a Whoosh full-text search index.

```yaml
nodes:
  search_index:
    type: WhooshStore
    config:
      persist_location: "/path/to/whoosh_index" # Optional: Index path
      # Additional Whoosh configuration options
```

**Input Ports:**
- `documents`: `List[Document]` - Documents to index

**Output Ports:**
- `status`: `str` - Indexing status message

#### ElasticsearchStore

Stores documents in an Elasticsearch cluster.

```yaml
nodes:
  es_store:
    type: ElasticsearchStore
    config:
      persist_location: "http://localhost:9200" # Required: Elasticsearch URL
      index_name: "my_documents"                # Optional: Index name
      # Additional Elasticsearch configuration options
```

**Input Ports:**
- `documents`: `List[Document]` - Documents to store

**Output Ports:**
- `status`: `str` - Storage status message

## Configuration Options

### Common Configuration Patterns

#### PDF Processing Options

```yaml
config:
  pdf_markdown: true          # Convert PDFs to markdown format
  pdf_unstructured: false     # Use unstructured parsing for complex PDFs
  infer_table_structure: true # Extract and preserve table structure
```

#### Metadata Enhancement

```yaml
config:
  store_md5: true            # Add MD5 hash to document metadata
  store_mimetype: true       # Add MIME type to document metadata  
  store_file_dates: true     # Add creation/modification dates
  extract_document_titles: true  # Extract document titles (requires LLM)
```

#### Performance Tuning

```yaml
config:
  n_proc: 4                  # Use 4 CPU cores for parallel processing
  verbose: true              # Show detailed progress information
  batch_size: 1000          # Process documents in batches
```

### Vector Store Configuration

Different storage backends support different configuration options:

```yaml
# ChromaDB
config:
  persist_location: "./chroma_db"
  collection_name: "documents"
  
# Whoosh
config:
  persist_location: "./whoosh_index"
  schema_fields: ["content", "title", "source"]

# Elasticsearch
config:
  persist_location: "https://elastic:password@localhost:9200"
  index_name: "document_index"
  basic_auth: ["username", "password"]
```

## Advanced Examples

### Multi-Source Pipeline

Process documents from multiple sources with different strategies:

```yaml
nodes:
  # Load PDFs with table extraction
  pdf_loader:
    type: LoadFromFolder
    config:
      source_directory: "pdfs/"
      pdf_markdown: true
      infer_table_structure: true
  
  # Load text files
  text_loader:
    type: LoadFromFolder
    config:
      source_directory: "texts/"
  
  # Chunk PDFs by paragraph (preserve structure)
  pdf_chunker:
    type: SplitByParagraph
    config:
      chunk_size: 1000
      chunk_overlap: 100
  
  # Chunk text files by character count
  text_chunker:
    type: SplitByCharacterCount
    config:
      chunk_size: 500
      chunk_overlap: 50
  
  # Store everything in unified index
  unified_store:
    type: WhooshStore
    config:
      persist_location: "unified_index"

connections:
  - from: pdf_loader
    from_port: documents
    to: pdf_chunker
    to_port: documents
  
  - from: text_loader
    from_port: documents
    to: text_chunker
    to_port: documents
  
  - from: pdf_chunker
    from_port: documents
    to: unified_store
    to_port: documents
  
  - from: text_chunker
    from_port: documents
    to: unified_store
    to_port: documents
```

### Document Processing Chain

Multiple processing steps in sequence:

```yaml
nodes:
  loader:
    type: LoadFromFolder
    config:
      source_directory: "documents/"
      extract_document_titles: true
  
  # First pass: large chunks for context
  coarse_chunker:
    type: SplitByParagraph
    config:
      chunk_size: 2000
      chunk_overlap: 200
  
  # Second pass: fine chunks for retrieval
  fine_chunker:
    type: SplitByCharacterCount
    config:
      chunk_size: 400
      chunk_overlap: 40
  
  # Store in vector database
  vector_store:
    type: ChromaStore
    config:
      persist_location: "vector_db"

connections:
  - from: loader
    from_port: documents
    to: coarse_chunker
    to_port: documents
  
  - from: coarse_chunker
    from_port: documents
    to: fine_chunker
    to_port: documents
  
  - from: fine_chunker
    from_port: documents
    to: vector_store
    to_port: documents
```

### Web Document Processing

Download and process documents from URLs:

```yaml
nodes:
  web_loader:
    type: LoadWebDocument
    config:
      url: "https://example.com/report.pdf"
      username: "api_user"
      password: "secret_key"
  
  doc_processor:
    type: SplitByCharacterCount
    config:
      chunk_size: 800
      chunk_overlap: 80
  
  search_store:
    type: ElasticsearchStore
    config:
      persist_location: "http://elasticsearch:9200"
      index_name: "web_documents"

connections:
  - from: web_loader
    from_port: documents
    to: doc_processor
    to_port: documents
  
  - from: doc_processor
    from_port: documents
    to: search_store
    to_port: documents
```

## Validation and Error Handling

The workflow engine performs comprehensive validation:

### Connection Validation

- **Port Existence**: Verifies source and target ports exist
- **Type Compatibility**: Ensures data types match between connections
- **Node Compatibility**: Enforces valid connection patterns:
  - ✅ Loader → TextSplitter
  - ✅ TextSplitter → TextSplitter  
  - ✅ TextSplitter → Storage
  - ❌ Loader → Storage (must have TextSplitter in between)
  - ❌ Storage → Any (Storage nodes are terminal)

### Runtime Validation

- **File Existence**: Checks that source directories and files exist
- **Configuration Validation**: Validates required parameters
- **Dependency Resolution**: Uses topological sorting to determine execution order
- **Cycle Detection**: Prevents infinite loops in workflows

### Error Messages

The engine provides detailed error messages:

```python
# Invalid node type
WorkflowValidationError: Unknown node type: InvalidNodeType

# Missing required configuration
NodeExecutionError: Node my_loader: source_directory is required

# Invalid connection
WorkflowValidationError: Loader node loader can only connect to TextSplitter nodes, not ChromaStoreNode

# Type mismatch
WorkflowValidationError: Type mismatch: loader.documents (List[Document]) -> storage.text (str)

# Missing port
WorkflowValidationError: Source node loader has no output port 'data'. Available: ['documents']
```

## Best Practices

### 1. Workflow Organization

```yaml
# Use descriptive node names
nodes:
  legal_document_loader:        # Not: loader1
    type: LoadFromFolder
    config:
      source_directory: "legal_docs/"
  
  contract_chunker:            # Not: splitter1
    type: SplitByParagraph
    config:
      chunk_size: 1500
  
  contract_search_index:       # Not: storage1
    type: WhooshStore
    config:
      persist_location: "contract_index"
```

### 2. Configuration Management

Use environment variables for paths and sensitive data:

```yaml
nodes:
  loader:
    type: LoadFromFolder
    config:
      source_directory: "${DOCUMENT_PATH:-./documents}"  # Default fallback
  
  es_store:
    type: ElasticsearchStore
    config:
      persist_location: "${ELASTICSEARCH_URL}"
      basic_auth: ["${ES_USERNAME}", "${ES_PASSWORD}"]
```

### 3. Chunk Size Guidelines

Choose appropriate chunk sizes for your use case:

```yaml
# Small chunks (200-400 chars) - Good for:
# - Precise retrieval
# - Question answering
# - Fine-grained search

# Medium chunks (500-1000 chars) - Good for:
# - General purpose RAG
# - Balanced context/precision
# - Most common use case

# Large chunks (1000-2000 chars) - Good for:
# - Document summarization
# - Context-heavy tasks
# - Preserving document structure
```

### 4. Performance Optimization

```yaml
nodes:
  bulk_loader:
    type: LoadFromFolder
    config:
      source_directory: "large_corpus/"
      n_proc: 8                    # Parallel processing
      verbose: false               # Reduce logging overhead
      batch_size: 500             # Process in batches
  
  efficient_chunker:
    type: SplitByCharacterCount
    config:
      chunk_size: 500
      chunk_overlap: 25           # Reduce overlap for speed
```

### 5. Metadata Preservation

```yaml
nodes:
  rich_loader:
    type: LoadFromFolder
    config:
      source_directory: "documents/"
      store_md5: true              # Document integrity
      store_mimetype: true         # File type tracking
      store_file_dates: true       # Temporal information
      extract_document_titles: true # Content-aware metadata
      infer_table_structure: true  # Preserve structure
```

## Troubleshooting

### Common Issues

#### 1. "Module not found" errors

```bash
# Ensure you're in the correct directory
cd /path/to/onprem/project

# Or add to Python path
export PYTHONPATH=/path/to/onprem:$PYTHONPATH
```

#### 2. "File not found" errors

```yaml
# Use absolute paths for reliability
nodes:
  loader:
    type: LoadFromFolder
    config:
      source_directory: "/full/path/to/documents"  # Not: "documents/"
```

#### 3. Memory issues with large documents

```yaml
# Process in smaller batches
nodes:
  loader:
    type: LoadFromFolder
    config:
      source_directory: "large_docs/"
      batch_size: 100              # Reduce batch size
      n_proc: 2                    # Reduce parallelism
  
  chunker:
    type: SplitByCharacterCount
    config:
      chunk_size: 300              # Smaller chunks
      chunk_overlap: 30            # Less overlap
```

#### 4. Storage connection issues

```yaml
# Test connectivity first
nodes:
  es_store:
    type: ElasticsearchStore
    config:
      persist_location: "http://localhost:9200"
      # Add authentication if needed
      basic_auth: ["username", "password"]
      # Increase timeouts if needed
      timeout: 30
```

### Debugging Workflows

Enable verbose output to see detailed execution:

```python
from onprem.pipelines.workflow import execute_workflow

# Detailed logging
results = execute_workflow("workflow.yaml", verbose=True)

# Check results
for node_id, result in results.items():
    print(f"Node {node_id}: {result}")
```

Validate before execution:

```python
from onprem.pipelines.workflow import WorkflowEngine

engine = WorkflowEngine()
try:
    engine.load_workflow_from_yaml("workflow.yaml")
    print("✓ Workflow validation passed")
except Exception as e:
    print(f"✗ Validation failed: {e}")
```

### Performance Monitoring

Track processing times and document counts:

```python
import time
from onprem.pipelines.workflow import execute_workflow

start_time = time.time()
results = execute_workflow("workflow.yaml", verbose=True)
end_time = time.time()

print(f"Processing time: {end_time - start_time:.2f} seconds")

# Count processed documents
total_docs = 0
for node_id, result in results.items():
    if 'documents' in result:
        count = len(result['documents'])
        print(f"{node_id}: {count} documents")
        total_docs += count

print(f"Total documents processed: {total_docs}")
```

---

This tutorial covers all aspects of the OnPrem workflow engine. For more examples, see the files in `nbs/tests/`. For API reference, check the source code in `onprem/pipelines/workflow.py`.