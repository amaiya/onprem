#!/usr/bin/env python3
"""
Simple standalone test for the new chat implementation.
Tests basic functionality without requiring pytest.
"""

import sys
import os

# Add onprem to path
sys.path.insert(0, os.path.abspath('.'))

def test_chatbot_structure():
    """Test that load_chatbot creates the expected structure"""
    print("Test 1: Chatbot structure...")
    from onprem import LLM

    llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)
    chatbot = llm.load_chatbot()

    # Should be a dict
    assert isinstance(chatbot, dict), f"Expected dict, got {type(chatbot)}"
    print("  ✓ chatbot is a dict")

    # Should have 'llm' and 'history' keys
    assert 'llm' in chatbot, "chatbot should have 'llm' key"
    print("  ✓ chatbot has 'llm' key")

    assert 'history' in chatbot, "chatbot should have 'history' key"
    print("  ✓ chatbot has 'history' key")

    # History should be a list
    assert isinstance(chatbot['history'], list), f"history should be a list, got {type(chatbot['history'])}"
    print("  ✓ history is a list")

    # History should start empty
    assert len(chatbot['history']) == 0, f"history should start empty, got {len(chatbot['history'])} items"
    print("  ✓ history starts empty")

    print("Test 1 PASSED\n")
    return True

def test_chat_basic():
    """Test basic chat functionality"""
    print("Test 2: Basic chat...")
    from onprem import LLM

    llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)

    # Make a chat call
    response = llm.chat("Say 'Hello World' and nothing else.")

    # Should return a string
    assert isinstance(response, str), f"Expected str, got {type(response)}"
    print(f"  ✓ Response is a string")

    # Should not be empty
    assert len(response) > 0, "Response should not be empty"
    print(f"  ✓ Response is not empty: '{response[:50]}...'")

    # Should contain expected content
    assert 'hello' in response.lower() or 'world' in response.lower(), f"Response should contain 'hello' or 'world', got: {response}"
    print(f"  ✓ Response contains expected content")

    print("Test 2 PASSED\n")
    return True

def test_history_grows():
    """Test that history grows with chat calls"""
    print("Test 3: History growth...")
    from onprem import LLM

    llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)
    chatbot = llm.load_chatbot()

    # Initial state
    assert len(chatbot['history']) == 0
    print("  ✓ History starts at 0")

    # First chat
    llm.chat("Hello")
    assert len(chatbot['history']) == 2, f"Expected 2 messages after 1 chat, got {len(chatbot['history'])}"
    print("  ✓ History has 2 messages after first chat (user + AI)")

    # Second chat
    llm.chat("How are you?")
    assert len(chatbot['history']) == 4, f"Expected 4 messages after 2 chats, got {len(chatbot['history'])}"
    print("  ✓ History has 4 messages after second chat")

    print("Test 3 PASSED\n")
    return True

def test_message_types():
    """Test that messages are proper LangChain objects"""
    print("Test 4: Message types...")
    from onprem import LLM
    from langchain_core.messages import HumanMessage, BaseMessage

    llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)
    llm.chat("Test message")

    chatbot = llm.load_chatbot()
    history = chatbot['history']

    # Should have 2 messages
    assert len(history) == 2
    print("  ✓ History has 2 messages")

    # First should be HumanMessage
    assert isinstance(history[0], (HumanMessage, BaseMessage))
    print("  ✓ First message is a Message object")

    # Second should be AIMessage
    assert isinstance(history[1], BaseMessage)
    print("  ✓ Second message is a Message object")

    # Check content
    assert hasattr(history[0], 'content')
    assert hasattr(history[1], 'content')
    print("  ✓ Messages have content attribute")

    print("Test 4 PASSED\n")
    return True

def test_conversation_memory():
    """Test that conversation context is maintained"""
    print("Test 5: Conversation memory...")
    from onprem import LLM

    llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)

    # Establish context
    response1 = llm.chat("My favorite color is blue. Remember this.")
    print(f"  Response 1: {response1[:50]}...")

    # Test recall
    response2 = llm.chat("What is my favorite color?")
    print(f"  Response 2: {response2[:50]}...")

    # Should remember the color
    assert 'blue' in response2.lower(), f"Model should remember 'blue', got: {response2}"
    print("  ✓ Model remembers previous context")

    print("Test 5 PASSED\n")
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("Testing new chat() implementation")
    print("="*60)
    print()

    tests = [
        ("Chatbot Structure", test_chatbot_structure),
        ("Basic Chat", test_chat_basic),
        ("History Growth", test_history_grows),
        ("Message Types", test_message_types),
        ("Conversation Memory", test_conversation_memory),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"Test '{name}' FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)

    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
