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
    
    # Load LLM to get model name
    llm = setup_llm()
    MODEL_NAME = llm.model_name
    
    # Add some CSS for better styling
    st.markdown("""
    <style>
    /* Improve chat message styling */
    .stChatMessage {
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Style the chat input */
    .stChatInputContainer {
        border-top: 1px solid #e0e0e0;
        padding-top: 1rem;
    }
    
    /* Make user messages stand out */
    .stChatMessage[data-testid="stChatMessage-user"] {
        background-color: #f0f7ff;
    }
    
    /* Style the assistant messages */
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background-color: #f9f9f9;
    }
    
    /* System messages should be subtle */
    .stChatMessage[data-testid="stChatMessage-system"] {
        background-color: #f0f0f0;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

    # Improved page header
    st.markdown("""
    <h1 style="
        color: #0068c9;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0068c9;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
    ">
        <span style="font-size: 1.8rem; margin-right: 0.5rem;">💬</span> 
        Interactive Chat
    </h1>
    """, unsafe_allow_html=True)
    
    # Enhanced model info card
    st.markdown(f"""
    <div style="
        padding: 12px 15px;
        border-radius: 8px;
        margin-bottom: 25px;
        background: linear-gradient(to right, rgba(0, 104, 201, 0.05), rgba(92, 137, 229, 0.1));
        border-left: 4px solid #0068c9;
        display: flex;
        align-items: center;
    ">
        <span style="font-size: 1.2rem; margin-right: 10px;">🤖</span>
        <div>
            <p style="margin: 0; font-weight: 500;">Current Model: <span style="color: #0068c9;">{MODEL_NAME}</span></p>
            <p style="margin: 3px 0 0 0; font-size: 0.85rem; color: #666;">
                This chat maintains conversation history for context-aware responses.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    init_chat_history()
    
    # Create a container for messages with fixed height and scrolling
    message_container = st.container()
    
    # Display chat messages with improved styling
    with message_container:
        for message in st.session_state.messages:
            # Choose appropriate avatars based on role
            if message["role"] == USER:
                avatar = "👤"
            elif message["role"] == ASSISTANT:
                avatar = "🤖"
            else:  # SYSTEM
                avatar = "ℹ️"
                
            # Display the message with its avatar
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    
    # Always display the chat input at the bottom
    chat_input = st.chat_input("Ask me anything...")
    
    # Determine the prompt to process
    if "example_to_process" in st.session_state and st.session_state.example_to_process:
        # When an example is clicked, use it as the prompt
        prompt = st.session_state.example_to_process
        # Clear it to prevent reprocessing
        st.session_state.example_to_process = None
    else:
        # Otherwise use normal chat input
        prompt = chat_input

    # Load LLM
    llm = setup_llm()

    
    # Process user input
    if prompt:
        # Add user message to chat
        add_message(USER, prompt)
        
        # Display the user message
        with st.chat_message(USER, avatar="👤"):
            st.markdown(prompt)
        
        # Generate LLM response with improved UI
        with st.chat_message(ASSISTANT, avatar="🤖"):
            message_placeholder = st.empty()
            
            # Show a typing indicator
            typing_indicator = st.empty()
            typing_indicator.markdown("<em>Thinking...</em>", unsafe_allow_html=True)
            
            with st.spinner(""):  # Hide the default spinner
                # Format prompt with history
                formatted_prompt = format_prompt_with_history(prompt)
                
                # Send to LLM
                response = ""
                try:
                    response = llm.prompt(formatted_prompt)
                except Exception as e:
                    response = f"⚠️ Error: {str(e)}"
                
                # Remove typing indicator and clear streamed output
                typing_indicator.empty()
                
                # Clear the streamed output from StreamHandler
                llm.llm.callbacks[0].container.empty()
                
                # Display final response
                message_placeholder.markdown(response)
        
        # Add assistant response to chat history
        add_message(ASSISTANT, response)
    
    # Create a nicer chat options section in the sidebar
    with st.sidebar:
        st.markdown("### Chat Options")
        
        # Add a primary-colored button to clear chat
        if st.button("🗑️ New Chat", type="primary", use_container_width=True):
            st.session_state.messages = [
                {"role": SYSTEM, "content": "Chat history cleared. How can I help you today?"}
            ]
            st.rerun()
        
        # Example prompts section
        st.markdown("### Try these examples")
        example_prompts = [
            "Explain how neural networks work",
            "Write a short poem about technology",
            "What are the best practices for code documentation?",
        ]
        
        # Create buttons for example prompts
        for i, example in enumerate(example_prompts):
            if st.button(f"💡 {example}", key=f"example_{i}", use_container_width=True):
                # Store the example in a session variable to be processed on the next rerun
                st.session_state.example_to_process = example
                
                # Force a rerun to process the example
                st.rerun()
        
        st.markdown("---")
        st.markdown(
            "*[More examples](https://amaiya.github.io/onprem/examples.html) of prompts*"
        )


if __name__ == "__main__":
    # Set page to wide mode when run directly
    st.set_page_config(
        page_title="Prompts", 
        page_icon="💬", 
        layout="centered",
        initial_sidebar_state="expanded"
    )
    main()
