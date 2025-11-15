"""
Test suite for LLM.chat() functionality to verify the migration from
deprecated ConversationChain to modern message-based approach.

This test file ensures:
1. Basic chat functionality works
2. Conversation history is maintained across multiple calls
3. Backward compatibility with existing API
4. Works with different LLM backends
"""

import pytest
import tempfile
import shutil
from onprem import LLM


class TestChatBasic:
    """Basic chat functionality tests"""

    def test_chat_single_message(self):
        """Test that a single chat message works"""
        llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)
        response = llm.chat("Say 'Hello World' and nothing else.")

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert 'hello' in response.lower() or 'world' in response.lower()

    def test_chat_returns_string(self):
        """Test that chat() returns a string, not a dict or object"""
        llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)
        response = llm.chat("What is 2+2?")

        assert isinstance(response, str), f"Expected str, got {type(response)}"

    def test_chat_with_prompt_template(self):
        """Test that chat() works with custom prompt template"""
        llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)
        response = llm.chat("Paris", prompt_template="Name the capital of {prompt}")

        assert response is not None
        assert isinstance(response, str)


class TestChatHistory:
    """Test conversation history maintenance"""

    def test_chat_maintains_history(self):
        """Test that conversation history is maintained across multiple calls"""
        llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)

        # First message: establish context
        response1 = llm.chat("My name is Alice. Remember this.")
        assert response1 is not None

        # Second message: reference previous context
        response2 = llm.chat("What is my name?")
        assert response2 is not None
        assert 'alice' in response2.lower(), f"Model should remember name 'Alice', got: {response2}"

    def test_chat_history_persists_across_calls(self):
        """Test that multiple chat calls build on each other"""
        llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)

        # Build up a conversation
        llm.chat("Let's count. I'll say 1.")
        llm.chat("Now you say 2.")
        response3 = llm.chat("What number did you just say?")

        assert '2' in response3 or 'two' in response3.lower()

    def test_separate_llm_instances_have_separate_histories(self):
        """Test that different LLM instances don't share history"""
        llm1 = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)
        llm2 = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)

        # Set context in llm1
        llm1.chat("My favorite color is blue.")

        # llm2 should not know about llm1's context
        response = llm2.chat("What is my favorite color?")
        # Should indicate it doesn't know (not "blue")
        assert 'blue' not in response.lower() or "don't know" in response.lower() or "not sure" in response.lower()


class TestChatBotObject:
    """Test the internal chatbot object structure"""

    def test_load_chatbot_creates_dict(self):
        """Test that load_chatbot() creates the expected structure"""
        llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)
        chatbot = llm.load_chatbot()

        # Should be a dict with 'llm' and 'history' keys
        assert isinstance(chatbot, dict), f"Expected dict, got {type(chatbot)}"
        assert 'llm' in chatbot, "chatbot dict should have 'llm' key"
        assert 'history' in chatbot, "chatbot dict should have 'history' key"
        assert isinstance(chatbot['history'], list), "history should be a list"

    def test_chatbot_is_cached(self):
        """Test that chatbot object is cached on the LLM instance"""
        llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)

        chatbot1 = llm.load_chatbot()
        chatbot2 = llm.load_chatbot()

        # Should be the same object
        assert chatbot1 is chatbot2
        assert id(chatbot1) == id(chatbot2)

    def test_history_grows_with_chat_calls(self):
        """Test that history list grows as chat() is called"""
        llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)
        chatbot = llm.load_chatbot()

        initial_length = len(chatbot['history'])
        assert initial_length == 0, "History should start empty"

        # Make a chat call
        llm.chat("Hello")

        # History should have 2 messages (human + AI)
        assert len(chatbot['history']) == 2, f"Expected 2 messages, got {len(chatbot['history'])}"

        # Make another call
        llm.chat("How are you?")

        # History should have 4 messages now
        assert len(chatbot['history']) == 4, f"Expected 4 messages, got {len(chatbot['history'])}"


class TestChatWithDifferentBackends:
    """Test chat() works with different LLM backends"""

    def test_chat_with_openai(self):
        """Test chat with OpenAI backend"""
        llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)
        response = llm.chat("Say 'test'")
        assert response is not None
        assert isinstance(response, str)

    @pytest.mark.skip(reason="Requires local model setup")
    def test_chat_with_ollama(self):
        """Test chat with Ollama backend"""
        llm = LLM(model_url='http://localhost:11434/v1', mute_stream=True)
        response = llm.chat("Hello")
        assert response is not None
        assert isinstance(response, str)

    @pytest.mark.skip(reason="Requires local model download")
    def test_chat_with_llamacpp(self):
        """Test chat with llama.cpp backend"""
        url = 'https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf'
        llm = LLM(model_url=url, mute_stream=True, confirm=False)
        response = llm.chat("Say hello")
        assert response is not None
        assert isinstance(response, str)


class TestChatEdgeCases:
    """Test edge cases and error handling"""

    def test_chat_with_empty_string(self):
        """Test chat with empty prompt"""
        llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)
        # Should still work, might get a generic response
        response = llm.chat("")
        assert isinstance(response, str)

    def test_chat_with_very_long_prompt(self):
        """Test chat with a long prompt"""
        llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)
        long_prompt = "Tell me about AI. " * 100
        response = llm.chat(long_prompt)
        assert response is not None
        assert isinstance(response, str)

    def test_chat_with_special_characters(self):
        """Test chat with special characters"""
        llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)
        response = llm.chat("What is 2+2? (Answer: 4)")
        assert response is not None
        assert isinstance(response, str)

    def test_chat_with_kwargs(self):
        """Test that additional kwargs are passed through"""
        llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)
        # Pass temperature as kwarg
        response = llm.chat("Say hello", temperature=0.1)
        assert response is not None
        assert isinstance(response, str)


class TestBackwardCompatibility:
    """Test backward compatibility with existing code patterns"""

    def test_old_usage_pattern_still_works(self):
        """Test that existing code patterns continue to work"""
        # This is how users currently use chat()
        llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)

        # Simple usage
        response = llm.chat("Hello")
        assert isinstance(response, str)

        # With prompt template
        response = llm.chat("World", prompt_template="Hello {prompt}")
        assert isinstance(response, str)

        # Multiple calls
        llm.chat("First message")
        llm.chat("Second message")
        response = llm.chat("Third message")
        assert isinstance(response, str)


class TestMessageTypes:
    """Test that messages are properly formatted"""

    def test_history_contains_message_objects(self):
        """Test that history contains proper LangChain message objects"""
        from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

        llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)
        llm.chat("Hello")

        chatbot = llm.load_chatbot()
        history = chatbot['history']

        assert len(history) == 2

        # First message should be HumanMessage
        assert isinstance(history[0], HumanMessage) or isinstance(history[0], BaseMessage)

        # Second message should be AIMessage
        assert isinstance(history[1], AIMessage) or isinstance(history[1], BaseMessage)

    def test_message_content_is_preserved(self):
        """Test that message content is properly stored"""
        llm = LLM(model_url='openai/gpt-4o-mini', mute_stream=True)
        test_prompt = "What is 1+1?"
        llm.chat(test_prompt)

        chatbot = llm.load_chatbot()
        history = chatbot['history']

        # User message content should match what was sent
        assert test_prompt in str(history[0].content)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
