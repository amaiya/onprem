import os
import sys
import streamlit as st
import time
from typing import List, Dict, Any

# Add parent directory to path to allow importing when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from parent modules
from webapp import read_config, DEFAULT_PROMPT
from utils import setup_llm, hide_webapp_sidebar_item

# Constants for message types
USER = "user"
ASSISTANT = "assistant"
SYSTEM = "system"

def init_chat_history():
    """Initialize chat history in session state if it doesn't exist"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": SYSTEM, "content": "Welcome! How can I help you today?"}
        ]

def add_message(role: str, content: str):
    """Add a message to the chat history"""
    st.session_state.messages.append({"role": role, "content": content, "time": time.time()})

def format_prompt_with_history(prompt: str) -> str:
    """Format the current prompt with conversation history for the LLM"""
    messages = []
    
    # Get all past messages except system messages and current prompt
    for msg in st.session_state.messages:
        if msg["role"] == SYSTEM:
            continue
        if msg["role"] == USER:
            messages.append(f"Human: {msg['content']}")
        elif msg["role"] == ASSISTANT:
            messages.append(f"Assistant: {msg['content']}")
    
    # Add the current prompt
    messages.append(f"Human: {prompt}")
    messages.append("Assistant:")
    
    # Join with double newlines for readability
    return "\n\n".join(messages)

def main():
    """
    Chat interface with memory stored in Streamlit session state
    """
    # Hide webapp sidebar item
    hide_webapp_sidebar_item()
    
    # Get configuration
    cfg = read_config()[0]
    PROMPT_TEMPLATE = cfg.get("prompt", {}).get("prompt_template", None)
    MODEL_NAME = os.path.basename(cfg.get("llm", {}).get("model_url", "UnknownModel"))
    
    # Page header
    st.header("💬 Chat")
    
    # Model info card
    st.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; margin-bottom: 20px; 
         background-color: rgba(0, 104, 201, 0.1); border-left: 4px solid #0068c9;">
      <p style="margin: 0;"><strong>Current Model:</strong> {MODEL_NAME}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    init_chat_history()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    prompt = st.chat_input("Type your message here...")
    
    # Load LLM
    llm = setup_llm()
    
    # Process user input
    if prompt:
        # Add user message to chat
        add_message(USER, prompt)
        
        # Display the user message
        with st.chat_message(USER):
            st.markdown(prompt)
        
        # Generate LLM response
        with st.chat_message(ASSISTANT):
            message_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                # Format prompt with history in a Human/Assistant format
                formatted_prompt = format_prompt_with_history(prompt)
                
                # Send to LLM (no prompt template to avoid conflicts with our formatting)
                response = ""
                try:
                    response = llm.prompt(formatted_prompt)
                except Exception as e:
                    response = f"Error generating response: {str(e)}"
                
                # Display the response
                message_placeholder.markdown(response)
        
        # Add assistant response to chat history
        add_message(ASSISTANT, response)
    
    # Add a clear chat button in an expander
    with st.expander("Chat Options", expanded=False):
        if st.button("Clear Chat History"):
            st.session_state.messages = [
                {"role": SYSTEM, "content": "Chat history cleared. How can I help you today?"}
            ]
            st.rerun()
            
        st.markdown("---")
        st.markdown(
            "*[Examples](https://amaiya.github.io/onprem/examples.html) of prompts for different problems*"
        )


if __name__ == "__main__":
    main()