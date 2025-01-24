"""
Run individual component tasks of WCST and LNT.
"""
import argparse
from pathlib import Path
from typing import Literal

from src.tests.wcst import WCST, WCSTConfig
from src.tests.lnt import LNT, LNTConfig
from src.models.openai import OpenAIWrapper, OpenAIConfig
from src.models.gemini import GeminiWrapper, GeminiConfig
from src.models.llama import LlamaWrapper, LlamaConfig

TaskType = Literal['shape', 'color', 'number', 'letter']

WCST_SHAPE_PROMPT = """
You are performing a card sorting task.
Match the card to the option that has the same shape.
Respond only with the number of the matching card.
"""

WCST_COLOR_PROMPT = """
You are performing a card sorting task.
Match the card to the option that has the same color.
Respond only with the number of the matching card.
"""

WCST_NUMBER_PROMPT = """
You are performing a card sorting task.
Match the card to the option that has the same number of shapes.
Respond only with the number of the matching card.
"""

LNT_LETTER_PROMPT = """
You are performing a letter classification task.
For each sequence, identify if the letter is a vowel or consonant.
Respond only with 'vowel' or 'consonant'.
"""

LNT_NUMBER_PROMPT = """
You are performing a number classification task.
For each sequence, identify if the number is even or odd.
Respond only with 'even' or 'odd'.
"""

def get_model(model_type: str, api_key: str):
    """Initialize appropriate model wrapper."""
    if model_type.startswith('gpt'):
        return OpenAIWrapper(api_key, OpenAIConfig(model=model_type))
    elif model_type.startswith('gemini'):
        return GeminiWrapper(api_key, GeminiConfig(model=model_type))
    elif model_type.startswith('llama'):
        return LlamaWrapper(api_key)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_task_prompt(task: TaskType) -> str:
    """Get system prompt for specific task."""
    prompts = {
        'shape': WCST_SHAPE_PROMPT,
        'color': WCST_COLOR_PROMPT,
        'number': WCST_NUMBER_PROMPT,
        'letter': LNT_LETTER_PROMPT
    }
    return prompts[task]

def run_component_task(
    model_type: str,
    api_key: str,
    task: TaskType,
    num_trials: int = 25
):
    """Run specific component task."""
    model = get_model(model_type, api_key)
    system_prompt = get_task_prompt(task)
    
    if task in ['shape', 'color', 'number']:
        test = WCST(WCSTConfig(num_trials=num_trials))
        test.current_rule = task  # Force specific rule
    else:
        test = LNT(LNTConfig(num_trials=num_trials))
        test.current_task = 'letter'

    for trial in range(num_trials):
        if task in ['shape', 'color', 'number']:
            card = test.deck[trial]
            options = test.generate_options(card)
            prompt = f"\nNew Card: {card}\n"
            for i, option in enumerate(options, 1):
                prompt += f"Option {i}: {option}\n"
            prompt += "Choose the correct option (1-4): "
        else:
            sequence = test.generate_sequence()
            prompt = f"\nSequence: {sequence}\n"

        response = model.send_message(prompt, system_prompt)
        print(f"Trial {trial + 1}:")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="API key")
    parser.add_argument("--model", required=True, help="Model type")
    parser.add_argument(
        "--task",
        required=True,
        choices=['shape', 'color', 'number', 'letter'],
        help="Component task to run"
    )
    parser.add_argument("--num-trials", type=int, default=25)
    args = parser.parse_args()

    run_component_task(args.model, args.api_key, args.task, args.num_trials)