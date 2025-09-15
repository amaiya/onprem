# Workflow Examples

This directory contains examples demonstrating the `onprem.pipelines.workflow` module.

## Files

- `example_workflow.yaml` - Simple workflow example
- `advanced_workflow_example.yaml` - More complex workflow with multiple sources
- `test_workflow_example.py` - Python test script demonstrating usage
- `workflow_examples_README.md` - This file

## Simple Example (`example_workflow.yaml`)

A basic 3-node pipeline:

```yaml
document_loader → text_chunker → search_index
```

1. **LoadFromFolder** - Loads documents from `sample_data/sotu`
2. **SplitByCharacterCount** - Chunks text (300 chars, 50 overlap)  
3. **WhooshStore** - Stores in search index at `test_whoosh_index`

## Advanced Example (`advanced_workflow_example.yaml`)

A more complex pipeline with multiple sources:

```yaml
sotu_loader → paragraph_chunker ↘
                                  unified_search_index
pdf_loader → character_chunker  ↗
```

1. **Two LoadFromFolder nodes** - Load from different directories
2. **Different chunking strategies** - Paragraphs vs character count
3. **Unified storage** - Both pipelines store to same WhooshStore

## Running the Examples

From the project root directory:

```bash
# Run the test script
python nbs/tests/test_workflow_example.py

# Or programmatically execute workflows:
from onprem.pipelines.workflow import execute_workflow
results = execute_workflow("nbs/tests/example_workflow.yaml")
```

## Available Node Types

### Loaders
- `LoadFromFolder` - Load documents from directory
- `LoadSingleDocument` - Load single file
- `LoadWebDocument` - Load from URL

### Text Splitters  
- `SplitByCharacterCount` - Chunk by character count
- `SplitByParagraph` - Chunk by paragraph (preserves structure)
- `KeepFullDocument` - No chunking (pass-through)

### Storage
- `WhooshStore` - Text search index
- `ChromaStore` - Vector database
- `ElasticsearchStore` - Hybrid search

## Node Configuration

Each node accepts configuration parameters:

```yaml
nodes:
  my_loader:
    type: LoadFromFolder
    config:
      source_directory: "/path/to/docs"
      verbose: true
      pdf_markdown: true  # Convert PDFs to markdown
      ignored_files: ["temp.txt"]
  
  my_chunker:
    type: SplitByCharacterCount  
    config:
      chunk_size: 500
      chunk_overlap: 50
      preserve_paragraphs: false
  
  my_store:
    type: WhooshStore
    config:
      persist_location: "/path/to/index"
```

## Validation

The workflow engine validates:

- Node types exist
- Connections are valid (correct ports and types)
- Node compatibility (Loader → TextSplitter → Storage)
- No cycles in the workflow graph

Invalid workflows will raise `WorkflowValidationError` with descriptive messages.