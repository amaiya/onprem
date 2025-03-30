# OnPrem.LLM Multipage Application

This is a multipage Streamlit application for OnPrem.LLM, providing a user-friendly interface to interact with local Large Language Models (LLMs).

## Pages

The application consists of multiple pages:

1. **Home** - Landing page with overview and quick navigation
2. **Prompts** - Use direct prompts to solve problems with the LLM
3. **Documents** - Talk to your documents using RAG (Retrieval-Augmented Generation)
4. **Search** - Search through your documents using keywords or semantic search
5. **Settings** - Configure application settings

## Running the Application

You can run the application using the `onprem` command:

```bash
onprem --port 8501 --address localhost
```

Or run it directly with Streamlit:

```bash
cd /path/to/onprem
streamlit run onprem/app/webapp.py
```

## Configuration

The application is configured using a YAML file located at:

```
{datadir}/webapp.yml
```

You can edit this file directly from the Settings page in the application.

## Structure

- `webapp.py` - Main entry point for the application
- `utils.py` - Utility functions shared across pages
- `pages/` - Directory containing individual page modules:
  - `0_Home.py` - Home page
  - `1_Prompts.py` - Prompts page
  - `2_Documents.py` - Documents page (RAG)
  - `4_Search.py` - Search page
  - `5_Settings.py` - Settings page