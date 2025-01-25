import pytest
import os
from chatbot.model import Model


class TestChatbotIntegration:
    IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

    @pytest.fixture(scope="module")
    def model(self):
        """Initialize model once for all tests"""
        model_id = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
        return Model(model_id)

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skip integ in Github Actions.")
    def test_end_to_end_response(self, model):
        """Test complete flow from prompt to response"""
        prompt = "What is Python?"
        response = model.generate_response(prompt)
        assert isinstance(response, str)
        assert len(response) > 0
