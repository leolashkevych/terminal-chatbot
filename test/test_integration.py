import pytest
from src.chatbot.model import Model
from src.chatbot.generator import ChatbotGenerator

class TestChatbotIntegration:
    @pytest.fixture(scope="module")
    def model(self):
        """Initialize model once for all tests"""
        model_id = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
        return Model(model_id)
    
    @pytest.fixture(scope="module")
    def generator(self, model):
        """Initialize generator once for all tests"""
        return ChatbotGenerator(model=model.model, tokenizer=model.tokenizer)

    def test_end_to_end_response(self, generator):
        """Test complete flow from prompt to response"""
        prompt = "What is Python?"
        response = generator.generate_response(prompt)
        assert isinstance(response, str)
        assert len(response) > 0

