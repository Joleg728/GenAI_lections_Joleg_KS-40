# llm_agent/core.py

import requests
import json
from typing import List, Dict, Optional
from decouple import config

from .tool_calculator import CalculatorTool
from .tool_websearch import WebSearchTool
from .tool_pdfinfo import PDFInfoTool
from .tool_passgen import PassGen

class LLMAgent:
    """
    LLM-агент, который планирует и выполняет задачи с помощью инструментов.
    Поддерживает как OpenRouter API, так и локальный Ollama.
    """

    def __init__(self, model: str = "tngtech/deepseek-r1t2-chimera", local: bool = False, 
                 ollama_base_url: str = "http://localhost:11434", ollama_model: str = "qwen3:0.6b"):
        """
        Инициализирует агента.
        
        Args:
            model (str): Название модели для OpenRouter.
            local (bool): Если True, использует локальный Ollama вместо OpenRouter.
            ollama_base_url (str): Базовый URL для Ollama API.
            ollama_model (str): Название модели в Ollama.
        """
        self.local = local
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        
        if not self.local:
            self.api_key = config('OPENROUTER_API_KEY')
            self.url = "https://openrouter.ai/api/v1/chat/completions"
            self.model = model
        else:
            self.api_key = None
            self.url = f"{self.ollama_base_url}/api/chat"
            self.model = ollama_model
        
        # Создаем экземпляры инструментов
        self.tools = {
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
            "pdf_info": PDFInfoTool(),
            "pass_gen": PassGen(),
        }
        self.conversation_history = []
    
    def _make_api_request(self, payload: Dict, headers: Optional[Dict] = None) -> Dict:
        """
        Универсальный метод для отправки запросов к API.
        Поддерживает как OpenRouter, так и Ollama.
        
        Args:
            payload (Dict): Тело запроса.
            headers (Dict, optional): Заголовки запроса.
            
        Returns:
            Dict: Ответ от API.
        """
        if headers is None:
            headers = {}
        
        if not self.local:
            headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
        else:
            headers["Content-Type"] = "application/json"
        
        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка при запросе к API: {e}")
    
    def _ask_llm_for_plan(self, query: str) -> List[Dict]:
        """
        Создает план действий, используя LLM.
        Работает как с OpenRouter, так и с Ollama.
        """
        system_prompt = """Available tools:
- pass_gen: for generating a NEW password. Use "params".
- web_search: ONLY for questions about news, facts, weather, current events.
- calculator: for math.
- pdf_info: for PDF files.

YOUR ONLY JOB is to output a JSON plan. You are NOT the assistant that answers the user.
You MUST NOT generate passwords, compute math, or answer the question yourself.
You only decide WHICH TOOLS to call.

Output format (strict):
{"plan": [ ... ]}

If the user asks to create a password, respond with:
{"plan": [{"action": "pass_gen", "params": {"length": <N>, "sp_symb": <bool>, "numbs": <bool>}}]}

CRITICAL RULES:
1. If the user asks to CREATE / GENERATE / MAKE a password
   (words: create, generate, make, password, пароль, сгенерируй, создай) — ALWAYS use pass_gen.
   NEVER use web_search for password generation, even if the request
   mentions length, character types, or security requirements.
2. Use web_search ONLY when the user asks a QUESTION about the world.
3. If unsure between pass_gen and web_search for the word "password" — choose pass_gen.
"""

        # Формируем запрос к API
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "stream": False,
        }

        if self.local:
            payload["think"] = False
            payload["format"] = "json"          # нативный Ollama: строка "json"
            payload["options"] = {"temperature": 0, "top_p": 0.1, "num_predict": 256}

        try:
            response_data = self._make_api_request(payload)

            if self.local:
                llm_text = response_data["message"]["content"]     # ← нативный путь
            else:
                llm_text = response_data["choices"][0]["message"]["content"]
            
            
            
            print("=" * 60)
            print("RAW LLM TEXT:")
            print(repr(llm_text))
            print("=" * 60)



            print(f">>>> RAW llm_text: {llm_text!r}")

            cleaned_json_text = self._extract_json(llm_text)
            print(f"> Ответ LLM для плана (очищенный): {cleaned_json_text!r}")

            action_plan = json.loads(cleaned_json_text)
            plan = action_plan.get("plan", [])
            return plan
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Произошла ошибка при создании плана: {e}")
            # Пробуем альтернативный подход: извлечь JSON из текста
            try:
                # Ищем JSON в тексте без маркеров
                import re
                json_match = re.search(r'\{.*"plan".*\}', llm_text, re.DOTALL)
                if json_match:
                    action_plan = json.loads(json_match.group())
                    return action_plan.get("plan", [])
            except:
                pass
            return []
            
    
    
    @staticmethod
    def _extract_json(text: str) -> str:
        """Достаёт JSON-объект из ответа LLM, чиня типичные артефакты."""
        import re

        text = text.strip()

        # 1. убрать markdown-обёртку, если есть
        fence = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()

        # 2. нормализовать похожие и невидимые юникод-символы
        text = (text
            .replace('\u00a0', ' ')   # NBSP
            .replace('\u200b', '')    # zero-width space
            .replace('\u200c', '')    # zero-width non-joiner
            .replace('\u200d', '')    # zero-width joiner
            .replace('\ufeff', '')    # BOM
            .replace('\u201c', '"').replace('\u201d', '"')  # “ ”
            .replace('\u2018', "'").replace('\u2019', "'")  # ‘ ’
            .replace('｛', '{').replace('｝', '}')
            .replace('［', '[').replace('］', ']')
            .replace('：', ':').replace('，', ',')
            .replace('＂', '"')
        )

        # 3. вырезать от первого '{' до последнего '}'
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"JSON-объект не найден в ответе LLM: {text!r}")

        return text[start:end + 1]
    
    

    def _generate_final_response(self, user_query: str) -> str:
        """
        Генерирует финальный ответ на основе истории выполнения.
        """
        prompt = f"""
        Based on the following conversation log, provide a direct and helpful answer to the user's original question.
        Be concise and use the information from the tool results to support your answer.

        Original User Question: {user_query}

        Conversation Log:
        {chr(10).join([msg['content'] for msg in self.conversation_history])}
        """
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if self.local:
            payload["stream"] = False
        try:
            response_data = self._make_api_request(payload)
            if self.local:
                final_text = response_data["message"]["content"]
            else:
                final_text = response_data["choices"][0]["message"]["content"]
            return final_text
        except Exception as e:
            return f"Ошибка при генерации финального ответа. Детали: {e}"

    def process_query(self, query: str) -> str:
        print(f"Агент анализирует ваш запрос... (Режим: {'локальный Ollama' if self.local else 'OpenRouter'})")
        
        # --- Шаг 1: Планирование ---
        plan = self._ask_llm_for_plan(query)

        if not plan:
            print("Инструменты не требуются. Генерирую ответ напрямую.")
            direct_prompt = f"Ответьте на следующий вопрос кратко и информативно: {query}"
            payload = {"model": self.model, "messages": [{"role": "user", "content": direct_prompt}]}
            if self.local:
                payload["stream"] = False
            try:
                response_data = self._make_api_request(payload)
                if self.local:
                    return response_data["message"]["content"]
                return response_data["choices"][0]["message"]["content"]
            except:
                return "Извините, не удалось сгенерировать ответ."

        # --- Шаг 2: Исполнение плана ---
        print(f"План действий: {plan}")
        last_result = None  # для сохранения результата последнего инструмента

        for step in plan:
            tool_name = step.get('action')
            tool_input = step.get('input')
            tool_params = step.get('params')

            if tool_name in self.tools:
                print(f"Выполняется инструмент: '{tool_name}'")
                try:
                    if tool_params is not None:
                        result = self.tools[tool_name].use(**tool_params)
                    elif tool_input is not None:
                        result = self.tools[tool_name].use(tool_input)
                    else:
                        result = "Ошибка: не указаны параметры"
                except Exception as e:
                    result = f"Ошибка при выполнении: {e}"
                print(f"Результат: {result}...")
                last_result = result  # сохраняем
                self.conversation_history.append({
                    'role': 'system',
                    'content': f"Tool {tool_name} result: {result}"
                })
            else:
                error_msg = f"Ошибка: инструмент '{tool_name}' не найден."
                print(error_msg)
                self.conversation_history.append({'role': 'system', 'content': error_msg})

        # --- Шаг 3: Если был вызван только pass_gen, сразу возвращаем его результат ---
        if len(plan) == 1 and plan[0].get('action') == 'pass_gen':
            return last_result

        # --- Шаг 4: Генерация финального ответа для остальных случаев ---
        print("Составляю финальный ответ...")
        final_response = self._generate_final_response(query)
        return final_response

    def test_ollama_connection(self) -> bool:
        """
        Тестирует соединение с локальным Ollama сервером.
        
        Returns:
            bool: True если соединение успешно, иначе False.
        """
        if not self.local:
            return False
        
        try:
            # Проверяем доступность Ollama API
            test_url = f"{self.ollama_base_url}/v1/models"
            response = requests.get(test_url)
            return response.status_code == 200
        except:
            return False
