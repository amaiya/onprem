# OnPrem.LLM Multipage Application

This is a multipage Streamlit application for OnPrem.LLM, providing a user-friendly interface to interact with local Large Language Models (LLMs).

## Pages

The application consists of multiple pages:

1. **Home** - Landing page with overview and quick navigation
2. **Prompts** - Use direct prompts to solve problems with the LLM
3. **Document QA** - Talk to your documents using RAG (Retrieval-Augmented Generation)
4. **Document Analysis** - Apply custom prompts to document chunks and export results
5. **Document Search** - Search through your documents using keywords or semantic search
6. **Manage** - Upload documents, manage folders, configure application settings

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
/<your home directory>/onprem_data/webapp/config.yml
```

You can edit this file directly from the Manage page in the application.
