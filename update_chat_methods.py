"""
Script to update the chat methods in the notebook to use modern LangChain patterns.
This replaces the deprecated ConversationChain with message-based history.
"""

import json
import sys

def update_notebook(notebook_path):
    """Update the notebook with new chat implementation"""

    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Find cell-4 which contains the LLM class
    target_cell = None
    cell_index = None

    for idx, cell in enumerate(notebook['cells']):
        if cell.get('id') == 'cell-4':
            target_cell = cell
            cell_index = idx
            break

    if not target_cell:
        print("ERROR: Could not find cell-4")
        return False

    # Get the source lines
    source_lines = target_cell['source']

    # Find and replace load_chatbot method
    load_chatbot_start = None
    load_chatbot_end = None

    for i, line in enumerate(source_lines):
        if 'def load_chatbot(self):' in line:
            load_chatbot_start = i
        if load_chatbot_start is not None and i > load_chatbot_start:
            # Look for the next method definition or end
            if line.strip().startswith('def ') and 'load_chatbot' not in line:
                load_chatbot_end = i
                break

    if load_chatbot_start is None:
        print("ERROR: Could not find load_chatbot method")
        return False

    # New load_chatbot implementation
    new_load_chatbot = [
        '    def load_chatbot(self):\n',
        '        """\n',
        '        Prepares chat functionality with message-based conversation history.\n',
        '        """\n',
        '        if self.chatbot is None:\n',
        '            llm = self.load_llm()\n',
        '\n',
        '            # Store chat history as list of messages\n',
        '            # Structure: {\'llm\': model, \'history\': [messages]}\n',
        '            self.chatbot = {\n',
        '                \'llm\': llm,\n',
        '                \'history\': []\n',
        '            }\n',
        '\n',
        '        return self.chatbot\n',
        '\n',
        '\n',
    ]

    # Replace load_chatbot
    source_lines[load_chatbot_start:load_chatbot_end] = new_load_chatbot

    # Now find and replace chat method
    chat_start = None
    chat_end = None

    for i, line in enumerate(source_lines):
        if 'def chat(self, prompt: str' in line:
            chat_start = i
        # Find the end of chat method (look for next method or end of class)
        if chat_start is not None and i > chat_start:
            # If we find a line that's not indented or is a new method at class level
            if line.strip() and not line.startswith('    '):
                chat_end = i
                break
            # Check if it's the end by looking for specific patterns
            if i < len(source_lines) - 1:
                next_line = source_lines[i + 1]
                if next_line.strip().startswith('show_doc') or next_line.strip().startswith('# |'):
                    chat_end = i + 1
                    break

    if chat_start is None:
        print("ERROR: Could not find chat method")
        return False

    # New chat implementation
    new_chat = [
        '    def chat(self, prompt: str, prompt_template=None, **kwargs):\n',
        '        """\n',
        '        Chat with LLM.\n',
        '\n',
        '        **Args:**\n',
        '\n',
        '        - *prompt*: a question you want to ask\n',
        '\n',
        '        """\n',
        '        from langchain_core.messages import HumanMessage\n',
        '\n',
        '        # Apply prompt template if provided\n',
        '        prompt_template = self.prompt_template if prompt_template is None else prompt_template\n',
        '        if prompt_template:\n',
        '            prompt = format_string(prompt_template, prompt=prompt)\n',
        '        \n',
        '        # Load chatbot (creates history if needed)\n',
        '        chatbot = self.load_chatbot()\n',
        '        \n',
        '        # Add user message to history\n',
        '        chatbot[\'history\'].append(HumanMessage(content=prompt))\n',
        '        \n',
        '        # Invoke LLM with full conversation history\n',
        '        response = chatbot[\'llm\'].invoke(chatbot[\'history\'], **kwargs)\n',
        '        \n',
        '        # Add AI response to history\n',
        '        chatbot[\'history\'].append(response)\n',
        '        \n',
        '        # Return response content\n',
        '        return response.content if hasattr(response, \'content\') else str(response)\n',
    ]

    # Replace chat method
    if chat_end:
        source_lines[chat_start:chat_end] = new_chat
    else:
        # If we couldn't find the end, just replace from start to end of list
        # This should not happen in a well-formed notebook
        print("WARNING: Could not determine end of chat method precisely")
        source_lines[chat_start:] = new_chat

    # Update the cell source
    target_cell['source'] = source_lines
    notebook['cells'][cell_index] = target_cell

    # Write back the notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"✓ Successfully updated {notebook_path}")
    print("  - Updated load_chatbot() to use message-based history")
    print("  - Updated chat() to work with new implementation")
    return True

if __name__ == '__main__':
    notebook_path = 'nbs/00_llm.base.ipynb'

    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]

    success = update_notebook(notebook_path)
    sys.exit(0 if success else 1)
