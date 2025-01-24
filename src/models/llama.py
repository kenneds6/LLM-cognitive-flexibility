"""
Meta Llama API wrapper for cognitive flexibility tests.
"""
import time
import re
from typing import Dict, List, Optional
from openai import OpenAI
from dataclasses import dataclass

@dataclass
class LlamaConfig:
    model: str = "meta-llama/Llama-3-70b"
    temperature: float = 0.7
    max_tokens: int = 100
    request_timeout: int = 30
    retry_delay: int = 1
    base_url: str = "https://api.deepinfra.com/v1/openai"

class LlamaWrapper:
    def __init__(self, api_key: str, config: LlamaConfig = LlamaConfig()):
        self.client = OpenAI(api_key=api_key, base_url=config.base_url)
        self.config = config
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
        """Send message to Llama API with retries."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": message})

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    stream=False
                )
                reply = response.choices[0].message.content
                self.conversation_history.extend([
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": reply}
                ])
                return reply
            except Exception as e:
                print(f"Error in API call: {e}")
                time.sleep(self.config.retry_delay)

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []