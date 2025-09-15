# OnPrem Workflow Tutorial

A comprehensive guide to using the OnPrem workflow engine for automated document processing pipelines.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Workflow Structure](#workflow-structure)
4. [Port Types and Data Flow](#port-types-and-data-flow)
5. [Node Types Reference](#node-types-reference)
6. [Configuration Options](#configuration-options)
7. [Advanced Examples](#advanced-examples)
8. [Validation and Error Handling](#validation-and-error-handling)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

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

## Port Types and Data Flow

The workflow system uses a **strongly-typed port system** to ensure data consistency and prevent invalid connections. Understanding port types is essential for building valid workflows.

### Port Type Overview

There are three main port types in the workflow system:

#### 📄 `List[Document]` - Document Collections
**Most common type** - Contains LangChain Document objects with content and metadata.

```python
# Document structure
Document(
    page_content="The actual text content of the document...",
    metadata={
        "source": "/path/to/file.pdf",
        "page": 1,
        "author": "John Smith",
        "extension": "pdf"
    }
)
```

**Used by:**
- All Loader → TextSplitter connections
- All TextSplitter → Storage connections  
- All Query → Processor connections

#### 📊 `List[Dict]` - Analysis Results
**Processing output** - Contains structured analysis results, summaries, or prompt responses.

```python
# Results structure
[
    {
        "document_id": 0,
        "source": "research.pdf",
        "prompt": "Analyze this document and provide...",
        "response": "TOPIC: AI Research | TECH: Neural networks | LEVEL: Advanced",
        "original_length": 1247,
        "metadata": {"page": 1, "author": "Smith", "year": 2023}
    }
]
```

**Used by:**
- All Processor → Exporter connections

#### ✅ `str` - Status Messages
**Completion status** - Simple text messages indicating operation results.

```python
# Status examples
"Successfully stored 150 documents in WhooshStore"
"Exported 25 results to analysis_report.xlsx"
"No documents to store"
```

**Used by:**
- Storage node outputs (terminal)
- Exporter node outputs (terminal)

### Data Flow Patterns

```
Raw Files → List[Document] → List[Document] → str
   ↓            ↓               ↓             ↓
Loader    TextSplitter      Storage      Status

Alternative analysis path:
Index → List[Document] → List[Dict] → str  
  ↓         ↓              ↓         ↓
Query   Processor      Exporter  Status
```

### Connection Validation

The workflow engine validates that connected ports have **matching types**:

✅ **Valid Connections:**
```yaml
# Document processing chain
- from: loader
  from_port: documents        # List[Document]
  to: chunker
  to_port: documents         # List[Document] ✓

# Analysis chain  
- from: query
  from_port: documents        # List[Document]
  to: processor
  to_port: documents         # List[Document] ✓

- from: processor
  from_port: results          # List[Dict]
  to: exporter
  to_port: results           # List[Dict] ✓
```

❌ **Invalid Connections:**
```yaml
# Type mismatch
- from: loader
  from_port: documents        # List[Document]
  to: exporter
  to_port: results           # List[Dict] ❌

# Wrong direction
- from: storage
  from_port: status           # str (terminal node)
  to: processor
  to_port: documents         # List[Document] ❌
```

### Port Naming Conventions

- **`documents`** - Always contains `List[Document]` objects
- **`results`** - Always contains `List[Dict]` with analysis results  
- **`status`** - Always contains `str` with completion messages

### Metadata Preservation

Data flows preserve metadata throughout the pipeline:

1. **Loader** → Document metadata (source, extension, dates, etc.)
2. **TextSplitter** → Preserves original metadata in chunks
3. **Query** → Returns documents with original metadata
4. **Processor** → Includes metadata in results under `metadata` key
5. **Exporter** → Flattens metadata into columns (`meta_source`, `meta_page`, etc.)

### Error Messages

When port types don't match, you'll see validation errors like:

```
WorkflowValidationError: Type mismatch: 
loader.documents (List[Document]) -> exporter.results (List[Dict])
```

```
WorkflowValidationError: Target node processor has no input port 'status'. 
Available: ['documents']
```

Understanding these port types helps you:
- Design valid workflows
- Debug connection errors
- Understand data transformations
- Plan processing pipelines

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
      ignored_files: ["temp.txt", "draft.doc"] # Optional: Specific files to skip
      include_patterns: ["*.pdf", "*.docx"]    # Optional: Only load files matching these patterns
      exclude_patterns: ["*draft*", "*temp*"]  # Optional: Skip files matching these patterns
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

**Filename Pattern Filtering:**

- `include_patterns`: Only process files matching these glob patterns (e.g., `["*.pdf", "*.doc*"]`)
- `exclude_patterns`: Skip files matching these glob patterns (e.g., `["*draft*", "*backup*"]`)
- If both specified, file must match include pattern AND not match exclude pattern
- Uses standard Unix glob patterns: `*` (any chars), `?` (single char), `[abc]` (character set)

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

### Query Nodes

Query nodes search existing storage indexes and return matching documents.

#### QueryWhooshStore

Searches documents in a Whoosh full-text search index.

```yaml
nodes:
  document_search:
    type: QueryWhooshStore
    config:
      persist_location: "/path/to/whoosh_index" # Required: Index path
      query: "artificial intelligence ML"        # Required: Search terms
      limit: 20                                 # Optional: Max results (default: 100)
```

**Input Ports:**
- None (queries existing storage directly)

**Output Ports:**
- `documents`: `List[Document]` - Matching documents

#### QueryChromaStore

Searches documents in a ChromaDB vector database using semantic similarity.

```yaml
nodes:
  vector_search:
    type: QueryChromaStore
    config:
      persist_location: "/path/to/chromadb"    # Required: Database path
      query: "machine learning algorithms"     # Required: Search query
      limit: 10                               # Optional: Max results (default: 10)
```

**Input Ports:**
- None (queries existing storage directly)

**Output Ports:**
- `documents`: `List[Document]` - Similar documents

### Processor Nodes

Processor nodes apply AI analysis, prompts, or transformations to documents.

#### PromptProcessor

Applies a custom prompt to each document using an LLM.

```yaml
nodes:
  document_analyzer:
    type: PromptProcessor
    config:
      prompt: |                               # Required: Prompt template
        Analyze this document and provide:
        1. Main topic: 
        2. Key findings:
        3. Complexity (1-5):
        
        Source: {source}
        Content: {content}
      model_name: "gpt-3.5-turbo"            # Optional: LLM model
      llm_type: "openai"                     # Optional: LLM provider
      batch_size: 5                          # Optional: Process in batches
```

**Prompt Variables:**
- `{content}` - Document content
- `{source}` - Document source path
- `{page}` - Page number (if available)
- Any metadata field (e.g., `{meta_author}`)

**Input Ports:**
- `documents`: `List[Document]` - Documents to process

**Output Ports:**
- `results`: `List[Dict]` - Analysis results with prompt responses

#### SummaryProcessor

Generates summaries for documents using an LLM.

```yaml
nodes:
  summarizer:
    type: SummaryProcessor
    config:
      max_length: 150                        # Optional: Max summary length in words
      model_name: "gpt-3.5-turbo"          # Optional: LLM model
      llm_type: "openai"                   # Optional: LLM provider
```

**Input Ports:**
- `documents`: `List[Document]` - Documents to summarize

**Output Ports:**
- `results`: `List[Dict]` - Summaries with metadata

### Exporter Nodes

Exporter nodes save processed results to various file formats.

#### CSVExporter

Exports results to CSV format for spreadsheet analysis.

```yaml
nodes:
  csv_output:
    type: CSVExporter
    config:
      output_path: "results.csv"            # Optional: Output file (default: results.csv)
      columns: ["source", "response"]       # Optional: Columns to include (default: all)
```

**Input Ports:**
- `results`: `List[Dict]` - Results to export

**Output Ports:**
- `status`: `str` - Export status message

#### ExcelExporter

Exports results to Excel format with formatting support.

```yaml
nodes:
  excel_output:
    type: ExcelExporter
    config:
      output_path: "analysis.xlsx"          # Optional: Output file (default: results.xlsx)
      sheet_name: "Document_Analysis"       # Optional: Sheet name (default: Results)
```

**Input Ports:**
- `results`: `List[Dict]` - Results to export

**Output Ports:**
- `status`: `str` - Export status message

#### JSONExporter

Exports results to JSON format for programmatic access.

```yaml
nodes:
  json_output:
    type: JSONExporter
    config:
      output_path: "results.json"           # Optional: Output file (default: results.json)
      pretty_print: true                    # Optional: Format JSON nicely (default: true)
```

**Input Ports:**
- `results`: `List[Dict]` - Results to export

**Output Ports:**
- `status`: `str` - Export status message

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

### Document Analysis Pipeline

Query existing storage, apply AI analysis, and export to spreadsheet:

```yaml
nodes:
  # Search for research documents
  research_query:
    type: QueryWhooshStore
    config:
      persist_location: "research_index"
      query: "methodology results conclusions findings"
      limit: 25
  
  # Analyze each document with custom prompts
  research_analysis:
    type: PromptProcessor
    config:
      prompt: |
        Analyze this research document and extract:
        
        1. RESEARCH QUESTION: What is the main research question?
        2. METHODOLOGY: What research methods were used?
        3. KEY FINDINGS: What are the 3 most important findings?
        4. LIMITATIONS: What limitations are mentioned?
        5. CONFIDENCE: How confident are the conclusions (High/Medium/Low)?
        
        Document: {source}
        Content: {content}
        
        Please format each answer on a separate line.
      model_name: "gpt-4"
      batch_size: 3
  
  # Export to Excel for review and analysis
  analysis_report:
    type: ExcelExporter
    config:
      output_path: "research_analysis_report.xlsx"
      sheet_name: "Document_Analysis"

connections:
  - from: research_query
    from_port: documents
    to: research_analysis
    to_port: documents
  
  - from: research_analysis
    from_port: results
    to: analysis_report
    to_port: results
```

### Multi-Format Export Pipeline

Process documents and export results in multiple formats:

```yaml
nodes:
  # Query for AI/ML papers
  ai_papers:
    type: QueryChromaStore
    config:
      persist_location: "vector_db"
      query: "artificial intelligence machine learning neural networks"
      limit: 15
  
  # Generate structured summaries
  paper_summaries:
    type: SummaryProcessor
    config:
      max_length: 200
      model_name: "gpt-3.5-turbo"
  
  # Export to CSV for spreadsheet analysis
  csv_export:
    type: CSVExporter
    config:
      output_path: "ai_paper_summaries.csv"
      columns: ["document_id", "source", "summary", "original_length"]
  
  # Export to JSON for programmatic access
  json_export:
    type: JSONExporter
    config:
      output_path: "ai_paper_summaries.json"
      pretty_print: true
  
  # Export to Excel for presentation
  excel_export:
    type: ExcelExporter
    config:
      output_path: "ai_paper_report.xlsx"
      sheet_name: "AI_ML_Summaries"

connections:
  # Single source, multiple outputs
  - from: ai_papers
    from_port: documents
    to: paper_summaries
    to_port: documents
  
  - from: paper_summaries
    from_port: results
    to: csv_export
    to_port: results
  
  - from: paper_summaries
    from_port: results
    to: json_export
    to_port: results
  
  - from: paper_summaries
    from_port: results
    to: excel_export
    to_port: results
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