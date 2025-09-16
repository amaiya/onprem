# OnPrem Workflow Engine

Create automated document processing pipelines using simple YAML configuration files. Instead of writing code, you visually design workflows by connecting different types of nodes.

## Table of Contents

1. [Quick Start - Three Core Examples](#quick-start---three-core-examples)
2. [Command-Line Usage](#command-line-usage)
3. [Workflow Structure](#workflow-structure)
4. [Port Types and Data Flow](#port-types-and-data-flow)
5. [Node Types Reference](#node-types-reference)
6. [Configuration Options](#configuration-options)
7. [Advanced Examples](#advanced-examples)
8. [Validation and Error Handling](#validation-and-error-handling)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Quick Start - Three Core Examples

The workflow engine provides three essential patterns that cover the most common document processing scenarios:

### 1. 🔄 Ingest PDFs to Vector Store ([`1_ingest_pdfs.yaml`](https://raw.githubusercontent.com/amaiya/onprem/refs/heads/master/nbs/tests/workflows/yaml_examples/1_ingest_pdfs.yaml))

**Purpose**: Load PDF files, chunk them, and store in a vector database for later retrieval.

```bash
# Run from the workflows directory
python -m onprem.workflow yaml_examples/1_ingest_pdfs.yaml
```

**What it does:**
- Loads PDF files from `../sample_data/`
- Converts PDFs to markdown for better processing
- Chunks documents into 800-character pieces with 80-character overlap
- Stores in ChromaDB vector database at `document_vectors/`

**Requirements**: PDF files in the sample_data directory

### 2. 🔍 Analyze from Vector Store ([`2_analyze_from_vectorstore.yaml`](https://raw.githubusercontent.com/amaiya/onprem/refs/heads/master/nbs/tests/workflows/yaml_examples/2_analyze_from_vectorstore.yaml))

**Purpose**: Query an existing vector database, apply AI analysis, and export results.

```bash
# Requires: Run example 1 first + set OPENAI_API_KEY
python -m onprem.workflow yaml_examples/2_analyze_from_vectorstore.yaml
```

**What it does:**
- Searches the vector database created in example 1
- Applies AI analysis to find documents about "artificial intelligence machine learning"
- Uses GPT-3.5-turbo to analyze each document for topic, key points, and relevance
- Exports results to `document_analysis_results.xlsx`

**Requirements**: 
- Run example 1 first to create `document_vectors/`
- Set `OPENAI_API_KEY` environment variable

### 3. 📄 Direct Document Analysis ([`3_direct_analysis.yaml`](https://raw.githubusercontent.com/amaiya/onprem/refs/heads/master/nbs/tests/workflows/yaml_examples/3_direct_analysis.yaml))

**Purpose**: Analyze documents directly without a vector database using two-stage AI processing.

```bash
# Requires: set OPENAI_API_KEY  
python -m onprem.workflow yaml_examples/3_direct_analysis.yaml
```

**What it does:**
- Loads documents directly from `../sample_data/`
- Processes complete documents (combines multi-page PDFs)
- Applies AI analysis to extract document type, topic, entities, summary, and recommendations
- Uses ResponseCleaner for consistent formatting
- Exports results to `document_analysis_summary.csv`

**Requirements**: 
- Documents in sample_data directory
- Set `OPENAI_API_KEY` environment variable

## Command-Line Usage

### Basic Execution
```bash
python -m onprem.workflow <workflow.yaml>
```

### Available Options
```bash
# Show help and examples
python -m onprem.workflow --help

# Validate workflow without running
python -m onprem.workflow --validate workflow.yaml

# List all available node types
python -m onprem.workflow --list-nodes

# Run quietly (suppress progress output)
python -m onprem.workflow --quiet workflow.yaml

# Show version
python -m onprem.workflow --version
```

### Example Commands
```bash
# Run the PDF ingestion workflow
python -m onprem.workflow yaml_examples/1_ingest_pdfs.yaml

# Validate a workflow before running
python -m onprem.workflow --validate yaml_examples/2_analyze_from_vectorstore.yaml

# See all available node types
python -m onprem.workflow --list-nodes
```

## Workflow Patterns

The three core examples demonstrate the main workflow patterns:

### Pattern 1: Data Ingestion (Example 1)
```
Documents → Chunking → Vector Store
```
Use this pattern to build searchable databases from your document collections.

### Pattern 2: Retrieval + Analysis (Example 2)
```
Vector Store → Query → AI Analysis → Export
```
Use this pattern to analyze specific topics from large document collections using semantic search.

### Pattern 3: Direct Processing (Example 3)
```  
Documents → Full Processing → AI Analysis → Cleanup → Export
```
Use this pattern for comprehensive analysis of entire document collections without intermediate storage.

## Available Node Types Summary

### 📁 Loaders
- **LoadFromFolder** - Load all documents from a directory
- **LoadSingleDocument** - Load a specific file
- **LoadWebDocument** - Download and load from URL

### ✂️ TextSplitters
- **SplitByCharacterCount** - Chunk by character count
- **SplitByParagraph** - Chunk by paragraphs (preserves structure)
- **KeepFullDocument** - Keep documents whole, optionally concatenate pages

### 🔧 DocumentTransformers
- **AddMetadata** - Add custom metadata fields to documents
- **ContentPrefix** - Prepend text to document content
- **ContentSuffix** - Append text to document content
- **DocumentFilter** - Filter documents by metadata or content criteria
- **PythonDocumentTransformer** - Custom Python transformations

### 🗄️ Storage  
- **ChromaStore** - Vector database for semantic search
- **WhooshStore** - Full-text search index
- **ElasticsearchStore** - Hybrid search capabilities

### 🔍 Query
- **QueryChromaStore** - Search vector database
- **QueryWhooshStore** - Search text index

### 🤖 Processors
- **PromptProcessor** - Apply AI analysis using custom prompts (DocumentProcessor)
- **ResponseCleaner** - Clean and format AI responses (ResultProcessor)
- **SummaryProcessor** - Generate document summaries (DocumentProcessor)

### 💾 Exporters
- **CSVExporter** - Export to CSV format
- **ExcelExporter** - Export to Excel format  
- **JSONExporter** - Export to JSON format

## Workflow Structure

### Basic Workflow YAML Format

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
from onprem.workflow import execute_workflow

# Execute the workflow
results = execute_workflow("my_workflow.yaml", verbose=True)
```

Or programmatically:

```python
from onprem.workflow import WorkflowEngine

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

Passes documents through without any chunking. Optionally concatenates multi-page documents and/or truncates documents to a maximum word count.

```yaml
nodes:
  no_split:
    type: KeepFullDocument
    config: {}  # No configuration needed - keeps documents as-is
  
  # For multi-page documents (PDFs, etc.) - combine into single document
  full_document:
    type: KeepFullDocument
    config:
      concatenate_pages: true    # Optional: Combine pages into single document
  
  # Truncate documents to first N words (useful for LLM context limits)
  truncated_document:
    type: KeepFullDocument
    config:
      max_words: 500            # Optional: Truncate to first 500 words
  
  # Both concatenation and truncation (applied in that order)
  combined_processing:
    type: KeepFullDocument
    config:
      concatenate_pages: true   # First: combine multi-page documents
      max_words: 1000          # Then: truncate to first 1000 words
```

**Page Concatenation:**

When `concatenate_pages: true`, multi-page documents are combined:
- Pages sorted by page number
- Content joined with `--- PAGE BREAK ---` separators
- Metadata preserved from first page plus additional fields:
  - `page: -1` (indicates full document)
  - `page_count: N` (number of pages combined)
  - `page_range: "1-5"` (original page range)
  - `concatenated: true` (flag indicating concatenation)

**Document Truncation:**

When `max_words: N` is specified, documents are truncated to the first N words:
- Word boundaries are preserved (no partial words)
- Metadata is enriched with truncation information:
  - `original_word_count: 2500` (original document length)
  - `truncated: true` (indicates truncation occurred)
  - `truncated_word_count: 500` (target truncation size)
- Documents shorter than `max_words` are passed through unchanged
- Processing order: concatenation first, then truncation

**Use Cases:**
- **Page Concatenation:**
  - Resume Processing - Combine multi-page resumes into single document
  - Contract Analysis - Process entire contracts as one unit
  - Report Analysis - Analyze complete reports without page boundaries
  - Legal Documents - Preserve document structure while enabling full-text analysis

- **Document Truncation:**
  - LLM Context Management - Fit long documents within token limits
  - Cost Control - Reduce processing costs for very long documents
  - Preview Generation - Create document summaries from beginnings
  - Performance Optimization - Speed up processing of large documents
  - Classification Tasks - Use document openings for categorization

**Input Ports:**
- `documents`: `List[Document]` - Documents to pass through or concatenate

**Output Ports:**
- `documents`: `List[Document]` - Unchanged or concatenated documents

### DocumentTransformer Nodes

DocumentTransformer nodes transform documents while preserving the `List[Document]` → `List[Document]` flow. They can add metadata, modify content, filter documents, or apply custom transformations. These nodes can be placed anywhere in the document pipeline.

#### AddMetadata

Adds static metadata fields to all documents for categorization and organization.

```yaml
nodes:
  categorize_meeting:
    type: AddMetadata
    config:
      metadata:
        category: "meeting20251001"
        department: "engineering"
        priority: "high"
        project: "Project Alpha"
        classification: "internal"
```

**Use Cases:**
- **Meeting Organization** - Tag all documents from a specific meeting
- **Project Tracking** - Add project identifiers to document collections
- **Department Categorization** - Organize documents by department or team
- **Classification** - Mark documents as confidential, internal, or public
- **Batch Processing** - Add consistent metadata to large document collections

**Input Ports:**
- `documents`: `List[Document]` - Documents to enrich

**Output Ports:**
- `documents`: `List[Document]` - Documents with added metadata

#### ContentPrefix

Prepends text to the page_content of all documents.

```yaml
nodes:
  mark_confidential:
    type: ContentPrefix
    config:
      prefix: "[CONFIDENTIAL - INTERNAL USE ONLY]"
      separator: "\n\n"  # Optional: separator between prefix and content (default: "\n\n")
  
  add_header:
    type: ContentPrefix
    config:
      prefix: "Project Alpha Documentation"
      separator: "\n---\n"
```

**Use Cases:**
- **Confidentiality Markings** - Add confidential headers to sensitive documents
- **Document Headers** - Add consistent headers to document collections
- **Processing Stamps** - Mark documents as processed by specific workflows
- **Context Addition** - Add contextual information to document beginnings

**Input Ports:**
- `documents`: `List[Document]` - Documents to modify

**Output Ports:**
- `documents`: `List[Document]` - Documents with prefixed content

#### ContentSuffix

Appends text to the page_content of all documents.

```yaml
nodes:
  add_footer:
    type: ContentSuffix
    config:
      suffix: |
        ---
        Document processed by OnPrem Workflow Engine
        Processing date: 2025-01-16
        For questions, contact: admin@company.com
      separator: "\n"  # Optional: separator between content and suffix (default: "\n\n")
```

**Use Cases:**
- **Processing Information** - Add processing timestamps and contact info
- **Legal Disclaimers** - Append legal text to documents
- **Document Footers** - Add consistent footers to document collections
- **Attribution** - Add source or processing attribution

**Input Ports:**
- `documents`: `List[Document]` - Documents to modify

**Output Ports:**
- `documents`: `List[Document]` - Documents with appended content

#### DocumentFilter

Filters documents based on metadata criteria, content patterns, or length requirements.

```yaml
nodes:
  filter_engineering:
    type: DocumentFilter
    config:
      # Filter by metadata
      metadata_filters:
        department: "engineering"
        status: "active"
      # Filter by content
      content_contains: ["project", "analysis", "results"]
      content_excludes: ["draft", "template"]
      # Filter by length
      min_length: 100
      max_length: 10000
  
  # Simple content filtering
  relevant_docs_only:
    type: DocumentFilter
    config:
      content_contains: ["machine learning", "AI", "neural network"]
      min_length: 50
```

**Filter Options:**
- `metadata_filters`: Dictionary of metadata key-value pairs that must match exactly
- `content_contains`: List of terms - document must contain at least one
- `content_excludes`: List of terms - document must not contain any
- `min_length`: Minimum content length in characters
- `max_length`: Maximum content length in characters

**Use Cases:**
- **Relevance Filtering** - Keep only documents containing specific keywords
- **Quality Control** - Remove documents that are too short or too long
- **Content Curation** - Filter out drafts, templates, or irrelevant content
- **Metadata-based Selection** - Keep only documents matching specific criteria

**Input Ports:**
- `documents`: `List[Document]` - Documents to filter

**Output Ports:**
- `documents`: `List[Document]` - Filtered documents

#### PythonDocumentTransformer

Executes custom Python code to transform documents with full flexibility and security controls.

```yaml
nodes:
  extract_document_info:
    type: PythonDocumentTransformer
    config:
      code: |
        # Available variables:
        # - doc: Document object
        # - content: doc.page_content (string)
        # - metadata: doc.metadata (mutable copy)
        # - document_id: index of document (int)
        # - source: source file path (string)
        
        # Extract information from content
        import re
        
        word_count = len(content.split())
        sentence_count = len(re.findall(r'[.!?]+', content))
        
        # Find email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)
        
        # Determine document type
        if 'meeting' in content.lower() and 'agenda' in content.lower():
            doc_type = 'meeting_agenda'
        elif 'analysis' in content.lower():
            doc_type = 'analysis_report'
        else:
            doc_type = 'general_document'
        
        # Enrich metadata
        metadata.update({
            'word_count': word_count,
            'sentence_count': sentence_count,
            'email_count': len(emails),
            'document_type': doc_type,
            'complexity_score': min(10, word_count // 100)
        })
        
        # Add summary to content
        summary = f"[{doc_type.upper()}: {word_count} words, Complexity: {metadata['complexity_score']}/10]"
        content = summary + "\n\n" + content
        
        # Create transformed document
        transformed_doc = Document(
            page_content=content,
            metadata=metadata
        )

  # Load transformation from external file
  complex_transform:
    type: PythonDocumentTransformer
    config:
      code_file: "scripts/document_enricher.py"
```

**Available Python Environment:**
- **Built-in functions**: `len`, `str`, `int`, `float`, `bool`, `list`, `dict`, `min`, `max`, etc.
- **Safe modules**: `re`, `json`, `math`, `datetime` (pre-imported)
- **Document class**: Available for creating new Document objects
- **Security**: No file I/O, network access, or system operations

**Variable Reference:**
- `doc`: Original Document object (read-only)
- `content`: Document content (modifiable string)
- `metadata`: Document metadata (modifiable dictionary copy)
- `document_id`: Index of current document (int)
- `source`: Source file path (string)
- `transformed_doc`: Set this to a Document object for the output (optional)

**Transformation Options:**
1. **Modify Variables**: Change `content` and `metadata`, let the system create the Document
2. **Explicit Creation**: Create and set `transformed_doc` explicitly

**Use Cases:**
- **Content Analysis** - Extract key information and add to metadata
- **Document Classification** - Automatically categorize documents by content
- **Data Extraction** - Find emails, URLs, phone numbers, etc.
- **Content Transformation** - Modify content based on complex rules
- **Custom Enrichment** - Add calculated metrics or derived information

**Input Ports:**
- `documents`: `List[Document]` - Documents to transform

**Output Ports:**
- `documents`: `List[Document]` - Transformed documents

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

Searches documents in a Whoosh full-text search index with support for different search types.

```yaml
nodes:
  # Sparse search (pure keyword matching)
  keyword_search:
    type: QueryWhooshStore
    config:
      persist_location: "/path/to/whoosh_index" # Required: Index path
      query: "artificial intelligence ML"        # Required: Search terms
      search_type: "sparse"                     # Optional: "sparse" or "semantic" (default: sparse)
      limit: 20                                 # Optional: Max results (default: 100)
      
  # Semantic search (keyword + embedding re-ranking)  
  smart_search:
    type: QueryWhooshStore
    config:
      persist_location: "/path/to/whoosh_index"
      query: "machine learning concepts"
      search_type: "semantic"                   # Uses keyword search + semantic re-ranking
      limit: 10
```

**Search Types:**
- `sparse`: Pure keyword/full-text search using Whoosh
- `semantic`: Keyword search followed by embedding-based re-ranking

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
      search_type: "semantic"                  # Optional: Only "semantic" supported (default)
      limit: 10                               # Optional: Max results (default: 10)
```

**Search Types:**
- `semantic`: Vector similarity search (only supported type)

**Input Ports:**
- None (queries existing storage directly)

**Output Ports:**
- `documents`: `List[Document]` - Similar documents

#### QueryElasticsearchStore

Searches documents in an Elasticsearch index with full support for all search types.

```yaml
nodes:
  # Sparse search (BM25 text matching)
  text_search:
    type: QueryElasticsearchStore
    config:
      persist_location: "http://localhost:9200" # Required: Elasticsearch URL
      index_name: "my_index"                    # Required: Index name
      query: "artificial intelligence"          # Required: Search query
      search_type: "sparse"                     # Optional: "sparse", "semantic", or "hybrid"
      limit: 5                                  # Optional: Max results (default: 10)
      
  # Semantic search (vector similarity)
  vector_search:
    type: QueryElasticsearchStore
    config:
      persist_location: "https://my-es:9200"
      index_name: "documents"
      query: "machine learning concepts" 
      search_type: "semantic"                   # Dense vector search
      limit: 3
      basic_auth: ["user", "password"]          # Optional: Authentication
      verify_certs: false                       # Optional: SSL verification
      
  # Hybrid search (combines text + vector)
  best_search:
    type: QueryElasticsearchStore
    config:
      persist_location: "http://localhost:9200"
      index_name: "knowledge_base"
      query: "deep learning neural networks"
      search_type: "hybrid"                     # Best of both worlds
      weights: [0.7, 0.3]                      # Optional: [text_weight, vector_weight]
      limit: 5
```

**Search Types:**
- `sparse`: Traditional BM25 text search
- `semantic`: Dense vector similarity search  
- `hybrid`: Weighted combination of sparse + semantic results

**Input Ports:**
- None (queries existing storage directly)

**Output Ports:**
- `documents`: `List[Document]` - Matching documents

### Processor Nodes

Processor nodes apply AI analysis, prompts, or transformations to documents.

#### PromptProcessor

Applies a custom prompt to each document using an LLM.

```yaml
nodes:
  document_analyzer:
    type: PromptProcessor
    config:
      prompt: |                               # Option 1: Inline prompt template
        Analyze this document and provide:
        1. Main topic: 
        2. Key findings:
        3. Complexity (1-5):
        
        Source: {source}
        Content: {content}
      llm:                                    # New flexible LLM configuration
        model_url: "openai://gpt-3.5-turbo"  # Model URL specification
        temperature: 0.7                     # Creativity level
        max_tokens: 1000                     # Response length limit
      batch_size: 5                          # Optional: Process in batches

  # Alternative: Load complex prompt from file with advanced LLM config
  complex_analyzer:
    type: PromptProcessor
    config:
      prompt_file: "prompts/statute_extraction.txt"  # Option 2: Load from file
      llm:                                   # Advanced LLM configuration
        model_url: "openai://gpt-4o-mini"    # Full model URL specification
        temperature: 0                       # Deterministic results
        mute_stream: true                    # Quiet processing
        timeout: 60                          # Request timeout
      batch_size: 2                          # Optional: Process in batches

```

**Loading Prompts from Files:**

For complex prompts, you can store them in separate text files and reference them with `prompt_file`:

```yaml
# File: prompts/resume_parser.txt
Analyze the resume and extract details in JSON format:
{
  "name": "...",
  "skills": ["...", "..."],
  "experience": [...]
}

Resume text: {content}

# Workflow configuration
config:
  prompt_file: "prompts/resume_parser.txt"
```

**Benefits of External Prompt Files:**
- Better organization for complex prompts
- Version control and collaboration
- Reusability across workflows
- Easier prompt engineering and testing

**LLM Configuration Options:**

The `llm` section accepts all parameters supported by the OnPrem LLM class.

**LLM Instance Sharing:**

The workflow engine automatically shares LLM instances between processors that use identical configurations, improving performance and memory usage:

```yaml
nodes:
  extractor:
    type: PromptProcessor
    config:
      prompt_file: "prompts/extraction.txt"
      llm:
        model_url: "openai://gpt-4o-mini"  # LLM instance created
        temperature: 0
        
  cleaner:
    type: ResponseCleaner  
    config:
      cleanup_prompt_file: "prompts/cleanup.txt"
      llm:
        model_url: "openai://gpt-4o-mini"  # Same instance reused!
        temperature: 0
```

**Configuration Options:**

```yaml
llm:
  # Model specification
  model_url: "openai://gpt-4o-mini"      # Model URL (recommended format)
  
  # Generation parameters
  temperature: 0.7                       # Randomness (0.0-2.0)
  max_tokens: 1500                       # Maximum response length
  top_p: 0.9                            # Nucleus sampling
  frequency_penalty: 0.0                 # Repetition penalty
  presence_penalty: 0.0                  # Topic diversity penalty
  
  # Behavior options
  mute_stream: true                      # Suppress streaming output
  timeout: 120                          # Request timeout in seconds
  
  # Provider-specific options (passed through)
  api_key: "${OPENAI_API_KEY}"          # API authentication
  base_url: "https://api.openai.com/v1"  # Custom API endpoint
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

#### ResponseCleaner

Post-processes and cleans LLM responses using another LLM call.

```yaml
nodes:
  response_cleaner:
    type: ResponseCleaner
    config:
      cleanup_prompt: |                  # Inline cleanup instructions
        Remove XML tags and clean up formatting while preserving all valid content:
        {original_response}
        
        Keep all important information, just remove formatting artifacts.
      llm:
        model_url: "openai://gpt-3.5-turbo"
        temperature: 0               # Deterministic cleanup

  # Alternative: Load cleanup prompt from file
  citation_cleaner:
    type: ResponseCleaner
    config:
      cleanup_prompt_file: "prompts/statute_cleanup.txt"  # Complex cleanup rules
      llm:
        model_url: "openai://gpt-4o-mini"
        temperature: 0
        mute_stream: true
```

**Use Cases:**
- **Extract structured data** from messy LLM responses
- **Remove formatting artifacts** like XML tags or unwanted text  
- **Clean statutory citations** while preserving all valid references (U.S.C., Public Laws, etc.)
- **Standardize outputs** for consistent data processing
- **Chain with PromptProcessor** for two-stage processing

**Important Notes:**
- Cleanup prompts should be carefully designed to avoid over-aggressive cleaning
- Always test with representative examples to ensure valid data isn't removed
- Consider the specific domain and format of your LLM responses

**Input Ports:**
- `results`: `List[Dict]` - Results from PromptProcessor to clean

**Output Ports:**
- `results`: `List[Dict]` - Cleaned results (original kept in `original_response` field)

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

#### PythonDocumentProcessor

Executes custom Python code on documents with proper security controls, allowing unlimited customization of document processing logic.

```yaml
nodes:
  # Inline Python code
  custom_analyzer:
    type: PythonDocumentProcessor
    config:
      code: |
        # Available variables:
        # - doc: Document object
        # - content: doc.page_content (string)
        # - metadata: doc.metadata (dict) 
        # - document_id: index of document (int)
        # - source: source file path (string)
        # - result: dictionary to populate (dict)
        
        # Extract key information (re module is pre-imported)
        word_count = len(content.split())
        sentence_count = len(re.findall(r'[.!?]+', content))
        
        # Find email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)
        
        # Populate result dictionary
        result['analysis'] = {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'emails_found': emails,
            'document_length': len(content)
        }
        result['processing_status'] = 'completed'

  # Load Python code from external file
  external_processor:
    type: PythonDocumentProcessor
    config:
      code_file: "scripts/document_analyzer.py"
```

**Available Python Environment:**
- **Built-in functions**: `len`, `str`, `int`, `float`, `bool`, `list`, `dict`, `min`, `max`, etc.
- **Safe modules**: `re`, `json`, `math`, `datetime` (pre-imported)
- **Document class**: Available for creating new Document objects
- **Security**: No file I/O, network access, or system operations

**Input Ports:**
- `documents`: `List[Document]` - Documents to process

**Output Ports:**
- `results`: `List[Dict]` - Processing results with custom analysis

#### PythonResultProcessor

Executes custom Python code on processing results, enabling post-processing and enhancement of analysis results.

```yaml
nodes:
  result_enhancer:
    type: PythonResultProcessor
    config:
      code: |
        # Available variables:
        # - result: original result dictionary (modifiable copy)
        # - original_result: read-only original result
        # - result_id: index of result (int) 
        # - processed_result: dictionary to populate (dict)
        
        # Enhance analysis results
        analysis = result.get('analysis', {})
        word_count = analysis.get('word_count', 0)
        
        # Categorize document by length
        if word_count < 100:
            category = 'short'
        elif word_count < 500:
            category = 'medium'
        else:
            category = 'long'
        
        # Create enhanced result
        processed_result['enhanced_analysis'] = {
            'original_analysis': analysis,
            'document_category': category,
            'complexity_score': min(10, word_count // 50),
            'has_emails': len(analysis.get('emails_found', [])) > 0
        }
        
        # Add summary
        processed_result['summary'] = f"Document categorized as '{category}'"
```

**Variable Naming Conventions:**
- **Document Processor**: Populate the `result` dictionary with your analysis
- **Result Processor**: Populate the `processed_result` dictionary with enhanced data

**Input Ports:**
- `results`: `List[Dict]` - Results to process

**Output Ports:**
- `results`: `List[Dict]` - Enhanced processing results

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

### Document Organization and Enrichment

Process documents with comprehensive metadata enrichment and content transformation:

```yaml
nodes:
  # Load meeting documents
  meeting_loader:
    type: LoadFromFolder
    config:
      source_directory: "meeting_docs/"
      include_patterns: ["*.pdf", "*.docx", "*.txt"]
      
  # Tag all documents with meeting metadata
  tag_meeting_info:
    type: AddMetadata
    config:
      metadata:
        meeting_id: "meeting20251001"
        department: "engineering"
        project: "Project Alpha"
        classification: "internal"
        attendees: "team_leads"
        
  # Add confidential header to all documents
  mark_confidential:
    type: ContentPrefix
    config:
      prefix: "[CONFIDENTIAL - PROJECT ALPHA TEAM ONLY]"
      separator: "\n\n"
      
  # Extract key information and enrich metadata
  analyze_content:
    type: PythonDocumentTransformer
    config:
      code: |
        # Analyze document content for key information
        import re
        
        # Basic statistics
        word_count = len(content.split())
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        # Look for action items and decisions
        action_items = len(re.findall(r'(?:action item|todo|task):', content, re.IGNORECASE))
        decisions = len(re.findall(r'(?:decision|resolved|agreed):', content, re.IGNORECASE))
        
        # Find mentions of team members
        team_members = re.findall(r'@(\w+)', content)
        
        # Classify document type
        content_lower = content.lower()
        if 'agenda' in content_lower:
            doc_type = 'meeting_agenda'
        elif action_items > 0 or 'action' in content_lower:
            doc_type = 'action_items'
        elif 'minutes' in content_lower or 'notes' in content_lower:
            doc_type = 'meeting_minutes'
        else:
            doc_type = 'meeting_document'
        
        # Update metadata with extracted information
        metadata.update({
            'word_count': word_count,
            'paragraph_count': paragraph_count,
            'action_items_count': action_items,
            'decisions_count': decisions,
            'mentioned_members': list(set(team_members)),
            'document_type': doc_type,
            'priority_score': min(10, (action_items * 2) + decisions),
            'has_action_items': action_items > 0,
            'complexity': 'high' if word_count > 1000 else 'medium' if word_count > 300 else 'low'
        })
        
        # Add document summary at the beginning
        summary = f"[{doc_type.upper()}: {word_count} words, {action_items} action items, Priority: {metadata['priority_score']}/10]"
        content = summary + "\n\n" + content
        
        # Create enriched document
        transformed_doc = Document(
            page_content=content,
            metadata=metadata
        )
        
  # Filter to keep only relevant documents
  filter_important:
    type: DocumentFilter
    config:
      metadata_filters:
        classification: "internal"
      # Keep documents with action items or decisions
      content_contains: ["action", "decision", "task", "todo"]
      min_length: 100
      
  # Add processing footer
  add_footer:
    type: ContentSuffix
    config:
      suffix: |
        
        ---
        Document processed: 2025-01-16
        Meeting ID: meeting20251001
        Next review: 2025-01-23
        Contact: project-alpha-admin@company.com
        
  # Chunk for storage
  chunk_docs:
    type: SplitByParagraph
    config:
      chunk_size: 1000
      chunk_overlap: 100
      
  # Store enriched documents
  meeting_store:
    type: WhooshStore
    config:
      persist_location: "meeting_20251001_index"

connections:
  - from: meeting_loader
    from_port: documents
    to: tag_meeting_info
    to_port: documents
    
  - from: tag_meeting_info
    from_port: documents
    to: mark_confidential
    to_port: documents
    
  - from: mark_confidential
    from_port: documents
    to: analyze_content
    to_port: documents
    
  - from: analyze_content
    from_port: documents
    to: filter_important
    to_port: documents
    
  - from: filter_important
    from_port: documents
    to: add_footer
    to_port: documents
    
  - from: add_footer
    from_port: documents
    to: chunk_docs
    to_port: documents
    
  - from: chunk_docs
    from_port: documents
    to: meeting_store
    to_port: documents
```

This example demonstrates the power of DocumentTransformer nodes:

1. **Metadata Tagging** - Organizes documents by meeting, project, and department
2. **Content Marking** - Adds confidential headers for security
3. **Intelligent Analysis** - Extracts action items, decisions, and team mentions
4. **Quality Filtering** - Keeps only documents with actionable content
5. **Processing Attribution** - Adds footer with processing information
6. **Searchable Storage** - Creates indexed, searchable document collection

The enriched metadata enables powerful queries like:
- "Find all documents from meeting20251001 with action items"
- "Show high-priority engineering documents from Project Alpha"
- "List all documents mentioning specific team members"

## Validation and Error Handling

The workflow engine performs comprehensive validation:

### Connection Validation

- **Port Existence**: Verifies source and target ports exist
- **Type Compatibility**: Ensures data types match between connections
- **Node Compatibility**: Enforces valid connection patterns:
  - ✅ Loader → TextSplitter
  - ✅ Loader → DocumentTransformer
  - ✅ TextSplitter → TextSplitter  
  - ✅ TextSplitter → DocumentTransformer
  - ✅ TextSplitter → Storage
  - ✅ DocumentTransformer → TextSplitter
  - ✅ DocumentTransformer → DocumentTransformer  
  - ✅ DocumentTransformer → Storage
  - ✅ Query → DocumentTransformer
  - ❌ Loader → Storage (must have TextSplitter or DocumentTransformer in between)
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

### Document Length Management

Control document size for LLM processing and cost optimization:

```yaml
nodes:
  # Load large documents
  document_loader:
    type: LoadFromFolder
    config:
      source_directory: "large_documents/"
  
  # Keep full documents but truncate to manageable size
  size_control:
    type: KeepFullDocument
    config:
      concatenate_pages: true    # First combine multi-page documents
      max_words: 2000           # Then truncate to first 2000 words
  
  # Analyze the controlled-size documents
  quick_analysis:
    type: PromptProcessor
    config:
      prompt: |
        Analyze this document excerpt (first 2000 words):
        
        Source: {source}
        Content: {content}
        
        Provide:
        1. Document type and purpose
        2. Main topics covered
        3. Key findings or conclusions
        4. Whether this appears to be the beginning, middle, or complete document
      llm:
        model_url: "openai://gpt-4o-mini"
        temperature: 0.1
        max_tokens: 500

connections:
  - from: document_loader
    from_port: documents
    to: size_control
    to_port: documents
    
  - from: size_control
    from_port: documents
    to: quick_analysis
    to_port: documents
```

**Benefits:**
- **Cost Control** - Process only document beginnings instead of entire files
- **Speed** - Faster analysis with consistent processing times
- **Context Management** - Ensure documents fit within LLM context windows
- **Preview Analysis** - Get quick insights from document openings
- **Metadata Tracking** - Know original document size and truncation status

### Debugging Workflows

Enable verbose output to see detailed execution:

```python
from onprem.workflow import execute_workflow

# Detailed logging
results = execute_workflow("workflow.yaml", verbose=True)

# Check results
for node_id, result in results.items():
    print(f"Node {node_id}: {result}")
```

Validate before execution:

```python
from onprem.workflow import WorkflowEngine

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
from onprem.workflow import execute_workflow

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

This tutorial covers all aspects of the OnPrem workflow engine.
