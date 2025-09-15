# Workflow Examples

This directory contains comprehensive examples and tools for the OnPrem workflow system.

## Quick Start

```bash
# List all available examples
python run_examples.py --list

# Run a basic example
python run_examples.py --run example_workflow.yaml

# Validate all workflows without running them
python run_examples.py --validate all

# Clean up output files
python run_examples.py --cleanup
```

## Files Overview

### 📚 Documentation
- `workflow_tutorial.md` - Complete tutorial with all node types and examples
- `workflow_examples_README.md` - Original examples documentation  
- `README.md` - This file

### 🚀 Example Workflows

| File | Difficulty | Description |
|------|------------|-------------|
| `example_workflow.yaml` | Beginner | Basic Load → Chunk → Store pipeline |
| `advanced_workflow_example.yaml` | Intermediate | Multi-source processing with different chunking |
| `pattern_filtering_examples.yaml` | Intermediate | Filename pattern filtering examples |
| `query_and_prompt_example.yaml` | Advanced | Query storage → Apply AI prompts → Export Excel |
| `complete_analysis_pipeline.yaml` | Advanced | Full pipeline with analysis and export |
| `simple_analysis_demo.yaml` | Advanced | Simple query → prompt → CSV export |

### 🛠️ Tools
- `run_examples.py` - Interactive example runner and validator
- `dev_testing_reference.py` - Development reference for testing new nodes

## Running Examples

### Prerequisites

1. **Sample Data**: Ensure `../sample_data/` exists with documents
2. **LLM Access**: Set API keys for advanced examples (OPENAI_API_KEY, etc.)
3. **Dependencies**: OnPrem package properly installed

### Basic Examples (No LLM Required)

```bash
# Simple document ingestion
python run_examples.py --run example_workflow

# Multi-source processing  
python run_examples.py --run advanced_workflow

# Pattern filtering (validate only - contains multiple workflows)
python run_examples.py --validate pattern_filtering_examples.yaml
```

### Advanced Examples (Require LLM)

```bash
# First create a search index with basic examples, then:
python run_examples.py --run query_and_prompt_example
python run_examples.py --run simple_analysis_demo
```

### Validation and Testing

```bash
# Check if all workflows are valid
python run_examples.py --validate all

# Dry run (validate without executing)  
python run_examples.py --dry-run complete_analysis_pipeline.yaml

# Clean up output files after testing
python run_examples.py --cleanup
```

## Example Output

### List Examples
```
$ python run_examples.py --list

📋 Available Workflow Examples:

 1. ✅ Basic Pipeline (Beginner)
     File: example_workflow.yaml
     Description: Simple Load → Chunk → Store pipeline using WhooshStore
     Requirements: sample_data/sotu directory

 2. ✅ Query & Prompt Analysis (Advanced)  
     File: query_and_prompt_example.yaml
     Description: Query storage index → Apply AI prompt → Export to Excel
     Requirements: existing_search_index/, LLM API key
     Note: Requires existing populated index and LLM
```

### Run Example
```
$ python run_examples.py --run example_workflow

🚀 Running workflow: example_workflow.yaml
Executing node: document_loader
  -> Processed 1 documents
Executing node: text_chunker  
  -> Processed 170 documents
Executing node: search_index
  -> Successfully stored 170 documents in WhooshStore

✅ Workflow completed successfully!

📁 Files/directories created:
   ✅ test_whoosh_index/
```

## Node Types Available

The workflow system supports these node categories:

- **Loaders** (3): `LoadFromFolder`, `LoadSingleDocument`, `LoadWebDocument`
- **TextSplitters** (3): `SplitByCharacterCount`, `SplitByParagraph`, `KeepFullDocument`  
- **Storage** (3): `ChromaStore`, `WhooshStore`, `ElasticsearchStore`
- **Query** (2): `QueryWhooshStore`, `QueryChromaStore`
- **Processors** (2): `PromptProcessor`, `SummaryProcessor`
- **Exporters** (3): `CSVExporter`, `ExcelExporter`, `JSONExporter`

## Common Issues

### Missing Sample Data
```bash
# Ensure sample data exists
ls ../sample_data/sotu/
ls ../sample_data/billionaires/
```

### LLM API Keys
```bash
# Set environment variables
export OPENAI_API_KEY="your-key-here"
```

### Missing Dependencies
```bash
# Install required packages
pip install pandas openpyxl  # For Excel export
```

## Creating Your Own Workflows

1. Start with `example_workflow.yaml` as a template
2. Refer to `workflow_tutorial.md` for complete node documentation
3. Use `run_examples.py --validate your_workflow.yaml` to check syntax
4. Test with `run_examples.py --dry-run your_workflow.yaml`

## Development

- See `dev_testing_reference.py` for testing new node types
- All workflows use the registry-based validation system
- New node types can be added by extending the base classes in `workflow.py`