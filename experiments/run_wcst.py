"""Wisconsin Card Sorting Test experiment runner."""
import os
import argparse
from typing import List, Dict
import json
from pathlib import Path
from datetime import datetime

from src.tests.wcst import WCST, WCSTConfig
from src.models.openai import OpenAIWrapper, OpenAIConfig
from src.models.gemini import GeminiWrapper, GeminiConfig
from src.models.llama import LlamaWrapper, LlamaConfig
from src.utils.logging import setup_logger

WCST_SYSTEM_PROMPT = """
You are participating in a card matching exercise.
For each trial, you will be presented with a card and four option cards.
Your task is to match the presented card with one of the options by responding with just the number (1-4).
There is always a correct way to match the cards, but you will need to discover it through trial and error.
When your match is correct, continue using the same matching approach until you receive feedback that it's incorrect.
When incorrect, you must switch to a completely different matching approach - do not persist with an approach that failed.
Respond only with a single number between 1 and 4.
Do not explain your choice or thought process.
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

def format_card(card: tuple) -> str:
    """Format card tuple as string."""
    return f"{card[0]} {card[1]} {card[2]}"

def run_experiment(
    model_type: str,
    api_key: str,
    num_evaluations: int = 8,
    config: WCSTConfig = WCSTConfig()
) -> List[Dict]:
    """Run multiple WCST evaluations and return results."""
    model = get_model(model_type, api_key)
    logger = setup_logger(model_type, "WCST")
    results = []

    for eval_num in range(num_evaluations):
        logger.info(f"Starting evaluation {eval_num + 1}/{num_evaluations}")
        test = WCST(config)
        model.reset_conversation()

        for trial in range(config.num_trials):
            card = test.deck[trial]
            options = test.generate_options(card)
            
            # Log the current card and all options
            logger.info(f"Trial {trial + 1}")
            logger.info(f"Current Card: {format_card(card)}")
            for i, option in enumerate(options, 1):
                logger.info(f"Option {i}: {format_card(option)}")
            
            prompt = f"\nNew Card: {format_card(card)}\n"
            for i, option in enumerate(options, 1):
                prompt += f"Option {i}: {format_card(option)}\n"
            prompt += "Choose the correct option (1-4): "
            
            response = model.send_message(prompt, WCST_SYSTEM_PROMPT)
            choice = model._extract_choice(response)
            
            if choice is None:
                logger.error(f"Invalid response format: {response}")
                continue
                
            is_correct = test.evaluate_choice(card, choice, options)
            feedback = "Correct!" if is_correct else "Incorrect!"
            
            logger.info(
                f"Chosen Option: Option {choice + 1} - {format_card(options[choice])}"
                f", Result={feedback}"
            )
            
            model.send_message(feedback)

        accuracy, score, trials = test.get_performance()
        eval_result = {
            "evaluation": eval_num + 1,
            "accuracy": accuracy,
            "score": score,
            "trials": trials
        }
        results.append(eval_result)
        logger.info(f"Evaluation {eval_num + 1} results: {eval_result}")

    return results

def save_results(results: List[Dict], model_name: str, output_dir: str = "results"):
    """Save experiment results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = output_path / f"wcst_{model_name}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="API key")
    parser.add_argument(
        "--model",
        required=True,
        help="Model type (gpt-3.5-turbo, gpt-4, gemini-1.5-pro, llama-70b)"
    )
    parser.add_argument("--num-evaluations", type=int, default=8)
    args = parser.parse_args()
    
    results = run_experiment(args.model, args.api_key, args.num_evaluations)
    save_results(results, args.model)