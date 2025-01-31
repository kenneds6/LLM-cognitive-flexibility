"""
Gemini API wrapper for cognitive flexibility tests.
"""
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import google.generativeai as genai

@dataclass
class GeminiConfig:
    model: str = "gemini-1.5-pro"
    temperature: float = 0.7
    max_tokens: int = 100
    request_timeout: int = 30
    retry_delay: int = 1

class GeminiWrapper:
    def __init__(self, api_key: str, config: GeminiConfig = GeminiConfig()):
        genai.configure(api_key=api_key)
        self.config = config
        self.model = genai.GenerativeModel(
            model_name=config.model,
            generation_config=genai.types.GenerationConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens
            )
        )
        self.conversation_history = []

    def _extract_choice(self, response: str) -> Optional[int]:
        """Extract numerical choice from response."""
        if "option" in response.lower():
            match = re.search(r'option\s?(\d+)', response, re.IGNORECASE)
            if match:
                return int(match.group(1)) - 1
        try:
            return int(response.strip()) - 1
        except ValueError:
            return None

    def _extract_ln_response(self, response: str) -> Optional[str]:
        """Extract letter-number task response."""
        matches = re.findall(r"vowel|consonant|even|odd", response.lower())
        return matches[0] if matches else None

    def send_message(self, message: str, system_prompt: str = "") -> str:
        """Send message to Gemini API."""
        # Combine system prompt and message
        full_prompt = system_prompt + "\n" + message if system_prompt else message

        try:
            chat_session = self.model.start_chat(
                history=self.conversation_history
            )
            response = chat_session.send_message(full_prompt)
            reply = response.text

            # Update conversation history
            self.conversation_history.extend([
                {"role": "user", "parts": [message]},
                {"role": "model", "parts": [reply]}
            ])

            return reply
        except Exception as e:
            print(f"Error in API call: {e}")
            return ""

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
