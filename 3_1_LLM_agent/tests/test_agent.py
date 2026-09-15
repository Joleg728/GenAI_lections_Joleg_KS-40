import pytest
import string
#from unittest.mock import MagicMock, patch
from llm_agent.core_v2 import LLMAgent

# =====================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ (Запускают реальную Ollama / API)
# =====================================================================
# Маркируем как 'integration', чтобы их можно было отключать при быстрой проверке

@pytest.mark.integration
def test_passgen_query_live():
    """Реальный запуск агента для проверки."""
    # Для тестов лучше использовать локальную модель, если она поднята
    agent = LLMAgent(local=True, ollama_model="qwen3.5:0.8b")
    query = "Create a password with thirteen characters, including special characters, but without numbers."
    
    response = agent.process_query(query)

    with open('example.txt', 'w', encoding='utf-8') as file:
        file.write(response)

    assert any(symb in response for symb in (string.ascii_letters + string.punctuation))