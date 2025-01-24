# Cognitive Flexibility of Large Language Models

This repository contains the implementation of experiments testing cognitive flexibility in Large Language Models (LLMs) using two neuropsychological tests:

1. Wisconsin Card Sorting Test (WCST) 
2. Letter-Number Test (LNT)

The methodology is described in detail in the paper:

> Kennedy, S. M., & Nowak, R. D. (2024). Cognitive flexibility of large language models. *ICML 2024 Workshop on LLMs and Cognition*. 

```bibtex
@inproceedings{kennedy2024cognitive,
  title={Cognitive flexibility of large language models},
  author={Kennedy, Sean M and Nowak, Robert D},
  booktitle={ICML 2024 Workshop on LLMs and Cognition},
  year={2024}
}
```

## Setup

1. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
2. Obtain API keys for the models you want to test:
   - OpenAI: Sign up at https://beta.openai.com/signup
   - Anthropic: Sign up at https://www.anthropic.com 
   - Add other models as needed
3. Set the API keys as environment variables or pass them as arguments to the scripts.

## Running Experiments

### Wisconsin Card Sorting Test
Run the WCST experiment on a specific model:
```
python experiments/run_wcst.py --api-key YOUR_API_KEY --model MODEL_NAME
```
Arguments:
- `--api-key`: Your API key for the selected model
- `--model`: Name of the model to test (e.g., `gpt-3.5-turbo`, `gpt-4`, `claude-v1`)
- `--num-evaluations` (optional): Number of times to run the test (default: 8)

### Letter Number Test
Run the LNT experiment on a specific model:
```
python experiments/run_lnt.py --api-key YOUR_API_KEY --model MODEL_NAME
``` 
Arguments:
- `--api-key`: Your API key for the selected model
- `--model`: Name of the model to test (e.g., `gpt-3.5-turbo`, `gpt-4`, `claude-v1`)
- `--num-evaluations` (optional): Number of times to run the test (default: 8)

## Analysis

### Analyzing Experiment Results
Analyze the results of the WCST and LNT experiments:
```
python analysis/analyze_results.py
```
This script loads the JSON result files, calculates summary statistics, and generates visualizations.

### Analyzing Component Task Performance
Analyze the performance on individual component tasks of WCST and LNT:
```
python analysis/component_analysis.py
```
This script evaluates model performance on shape matching, color matching, number matching (WCST) and letter classification, number parity (LNT).

## Adding New Models

To add support for a new LLM:
1. Create a new wrapper class in `src/models/` 
2. Implement the required methods:
   - `send_message(message: str, system_prompt: str) -> str` 
   - `reset_conversation()`
3. Update the `get_model()` function in `run_wcst.py` and `run_lnt.py` to include the new model.

## Reproducing the Experiments

To reproduce the experiments and analysis from the paper:
1. Set up the environment as described in the Setup section
2. Run the WCST and LNT experiments on the desired models
3. Run the analysis scripts to process the results
4. Compare the results to those reported in the paper

Refer to the configuration files in `config/` to see the specific parameters used in the paper.

Please open an issue if you encounter any problems reproducing the results. Contributions are welcome!

## License

MIT