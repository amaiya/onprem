# OnPrem Workflow Engine

The OnPrem workflow engine allows you to create automated document processing pipelines using simple YAML configuration files. Instead of writing code, you visually design workflows by connecting different types of nodes.

## Quick Start - Three Core Examples

The workflow engine provides three essential patterns that cover the most common document processing scenarios:

### 1. 🔄 Ingest PDFs to Vector Store (`1_ingest_pdfs.yaml`)

**Purpose**: Load PDF files, chunk them, and store in a vector database for later retrieval.

```bash
# Run from the workflows directory
python -m onprem.pipelines.workflow yaml_examples/1_ingest_pdfs.yaml
```

**What it does:**
- Loads PDF files from `../sample_data/`
- Converts PDFs to markdown for better processing
- Chunks documents into 800-character pieces with 80-character overlap
- Stores in ChromaDB vector database at `document_vectors/`

**Requirements**: PDF files in the sample_data directory

### 2. 🔍 Analyze from Vector Store (`2_analyze_from_vectorstore.yaml`)

**Purpose**: Query an existing vector database, apply AI analysis, and export results.

```bash
# Requires: Run example 1 first + set OPENAI_API_KEY
python -m onprem.pipelines.workflow yaml_examples/2_analyze_from_vectorstore.yaml
```

**What it does:**
- Searches the vector database created in example 1
- Applies AI analysis to find documents about "artificial intelligence machine learning"
- Uses GPT-3.5-turbo to analyze each document for topic, key points, and relevance
- Exports results to `document_analysis_results.xlsx`

**Requirements**: 
- Run example 1 first to create `document_vectors/`
- Set `OPENAI_API_KEY` environment variable

### 3. 📄 Direct Document Analysis (`3_direct_analysis.yaml`)

**Purpose**: Analyze documents directly without a vector database using two-stage AI processing.

```bash
# Requires: set OPENAI_API_KEY  
python -m onprem.pipelines.workflow yaml_examples/3_direct_analysis.yaml
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
python -m onprem.pipelines.workflow <workflow.yaml>
```

### Available Options
```bash
# Show help and examples
python -m onprem.pipelines.workflow --help

# Validate workflow without running
python -m onprem.pipelines.workflow --validate workflow.yaml

# List all available node types
python -m onprem.pipelines.workflow --list-nodes

# Run quietly (suppress progress output)
python -m onprem.pipelines.workflow --quiet workflow.yaml

# Show version
python -m onprem.pipelines.workflow --version
```

### Example Commands
```bash
# Run the PDF ingestion workflow
python -m onprem.pipelines.workflow yaml_examples/1_ingest_pdfs.yaml

# Validate a workflow before running
python -m onprem.pipelines.workflow --validate yaml_examples/2_analyze_from_vectorstore.yaml

# See all available node types
python -m onprem.pipelines.workflow --list-nodes
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

## Available Node Types

### 📁 Loaders
- **LoadFromFolder** - Load all documents from a directory
- **LoadSingleDocument** - Load a specific file
- **LoadWebDocument** - Download and load from URL

### ✂️ TextSplitters
- **SplitByCharacterCount** - Chunk by character count
- **SplitByParagraph** - Chunk by paragraphs (preserves structure)
- **KeepFullDocument** - Keep documents whole, optionally concatenate pages

### 🗄️ Storage  
- **ChromaStore** - Vector database for semantic search
- **WhooshStore** - Full-text search index
- **ElasticsearchStore** - Hybrid search capabilities

### 🔍 Query
- **QueryChromaStore** - Search vector database
- **QueryWhooshStore** - Search text index

### 🤖 Processors
- **PromptProcessor** - Apply AI analysis using custom prompts
- **ResponseCleaner** - Clean and format AI responses
- **SummaryProcessor** - Generate document summaries

### 💾 Exporters
- **CSVExporter** - Export to CSV format
- **ExcelExporter** - Export to Excel format  
- **JSONExporter** - Export to JSON format

## Example Helper Scripts

### List All Examples
```bash
python run_examples.py --list
```

### Validate All Workflows
```bash
python run_examples.py --validate all
```

### Run Specific Example
```bash
python run_examples.py --run 1_ingest_pdfs.yaml
```

## Troubleshooting

### Common Issues

**File not found errors:**
- Use absolute paths or ensure you're in the correct directory
- Check that `../sample_data/` exists and contains documents

**API key errors:**
- Set your OpenAI API key: `export OPENAI_API_KEY="your-key-here"`
- Examples 2 and 3 require LLM access

**Module import errors:**
- Ensure you're running from the project root or have the correct PYTHONPATH
- Install OnPrem with: `pip install onprem[chroma]`

**Validation errors:**
- Use `--validate` to check workflows before running
- Check that all required configuration parameters are provided

### Getting Help

```bash
# Show detailed help
python -m onprem.pipelines.workflow --help

# Validate before running
python -m onprem.pipelines.workflow --validate your_workflow.yaml

# List examples with requirements
python run_examples.py --list
```

## Next Steps

1. **Start with Example 1** - Build your first document database
2. **Try Example 2** - Learn retrieval-based analysis  
3. **Explore Example 3** - Master direct document processing
4. **Check the Archive** - See advanced patterns in `yaml_examples/archive/`
5. **Read the Tutorial** - Full documentation in `workflow_tutorial.md`

The three core examples provide a complete learning path from basic ingestion to advanced AI-powered analysis!