import os
import sys
import streamlit as st

# Add parent directory to path to allow importing when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from parent modules
from webapp import read_config, DEFAULT_PROMPT
from utils import setup_llm, hide_webapp_sidebar_item

def main():
    """
    Page for using prompts to solve problems
    """
    # Hide webapp sidebar item
    hide_webapp_sidebar_item()
    
    cfg = read_config()[0]
    PROMPT_TEMPLATE = cfg.get("prompt", {}).get("prompt_template", None)
    
    st.header("Use Prompts to Solve Problems")
    
    prompt = st.text_area(
        "Submit a Prompt to the LLM:",
        "",
        height=100,
        placeholder=DEFAULT_PROMPT,
        help="Tip: If you don't like the response quality after pressing 'Submit', try pressing the button a second time. "
        "You can also try re-phrasing the prompt.",
    )
    submit_button = st.button("Submit")
    st.markdown(
        "*Examples of using prompts to solve different problems are [here](https://amaiya.github.io/onprem/examples.html).*"
    )
    st.markdown("---")
    
    llm = setup_llm()
    if prompt and submit_button:
        saved_output = llm.prompt(prompt, prompt_template=PROMPT_TEMPLATE)


if __name__ == "__main__":
    main()