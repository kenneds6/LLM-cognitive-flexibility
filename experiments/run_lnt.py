"""Letter Number Test experiment runner."""
import os
import argparse
from typing import List, Dict
import json
from pathlib import Path
from datetime import datetime

from src.tests.lnt import LNT, LNTConfig
from src.models.openai import OpenAIWrapper, OpenAIConfig
from src.models.gemini import GeminiWrapper, GeminiConfig
from src.models.llama import LlamaWrapper, LlamaConfig
from src.utils.logging import setup_logger

LNT_SYSTEM_PROMPT = """
You are participating in a sequence classification exercise.
For each trial, you will see a sequence containing one letter followed by one number.
Your task is to classify the sequence in one of two ways:
For letters: respond with 'vowel' or 'consonant'
For numbers: respond with 'even' or 'odd'
You must choose ONE type of classification and stick with it while it works.
If you receive incorrect feedback, you must switch to the other classification task - do not persist with a failed approach.
Respond only with a single word: 'vowel', 'consonant', 'even', or 'odd'.
Do not explain your choice or provide both classifications.
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

def run_experiment(
    model_type: str,
    api_key: str,
    num_evaluations: int = 8,
    config: LNTConfig = LNTConfig()
) -> List[Dict]:
    """Run multiple LNT evaluations and return results."""
    model = get_model(model_type, api_key)
    logger = setup_logger(model_type, "LNT")
    results = []

    for eval_num in range(num_evaluations):
        logger.info(f"Starting evaluation {eval_num + 1}/{num_evaluations}")
        test = LNT(config)
        model.reset_conversation()

        for trial in range(config.num_trials):
            sequence = test.generate_sequence()
            prompt = f"\nSequence: {sequence}\n"
            
            response = model.send_message(prompt, LNT_SYSTEM_PROMPT)
            choice = model._extract_ln_response(response)
            
            if choice is None:
                logger.error(f"Invalid response format: {response}")
                continue
                
            is_correct = test.evaluate_response(sequence, choice)
            feedback = "Correct!" if is_correct else "Incorrect!"
            
            logger.info(
                f"Trial {trial + 1}: Sequence={sequence}, "
                f"Response={choice}, Result={feedback}"
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
    filename = output_path / f"lnt_{model_name}_{timestamp}.json"
    
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