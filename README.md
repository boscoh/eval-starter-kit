# AI Eval Starter Kit

A ready-to-use toolkit for quickly setting up and running evaluations of AI model responses. This starter kit provides pre-configured evaluation metrics and templates to help you assess and compare AI model outputs with minimal setup.

## Features

- 🎯 **Multiple Evaluation Metrics**:
  - Coherence evaluation
  - Answer equivalence checking
  - Word count validation
- 🔄 **Model Agnostic**: Works with both local (Ollama) and cloud (OpenAI) models
- ⚙️ **Configuration Driven**: Simple YAML-based test configuration
- 📊 **Statistical Analysis**: Provides mean and standard deviation for repeated evaluations
- 🏗️ **Extensible Architecture**: Easy to add new evaluation metrics

## Quick Start

### Prerequisites
- Python 3.8+
- `uv` package manager (`pip install uv`)
- One of the following:
  - For local models: [Ollama](https://ollama.ai/) installed and running
  - For OpenAI: `OPENAI_API_KEY` set in your environment or `.env` file

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd eval
   ```

2. Install dependencies:
   ```bash
   # Install uv if not already installed
   curl -sSf https://astral.sh/uv/install.sh | sh
   
   uv install -r requirements.txt
   ```

### Running Evaluations

1. **Using Ollama (Local Models)**
   ```bash
   # Start Ollama server (if not already running)
   ollama serve
   
   # Run an evaluation
   uv run eval_runner.py sample-evals/engineer.yaml
   ```

2. **Using OpenAI**
   ```bash
   # Set your OpenAI API key
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   
   # Run an evaluation
   uv run eval_runner.py sample-evals/engineer.yaml --service openai --model gpt-4
   ```

## Configuration

Evaluation configurations are defined in YAML files. See the `SampleEvals` directory for examples.

### Example Configuration

```yaml
# sample-evals/engineer.yaml
name: "Engineering Candidate Evaluation"
system_prompt_ref: "candidate-skills"  # References a file in system-prompts/
service: "ollama"  # or "openai"
model: "llama3.2"  # or "gpt-4", etc.
prompt: |
  Please analyze the following resume and provide a summary of the candidate's 
  technical skills and experience.
  
  [RESUME CONTENT HERE]

evaluators:
  - CoherenceEvaluator
  - WordCountEvaluator:
      min_words: 100
      max_words: 300
```

## Available Evaluators

1. **CoherenceEvaluator**
   - Evaluates the logical flow and coherence of the response
   - Returns a score between 0.0 (incoherent) and 1.0 (highly coherent)

2. **EquivalenceEvaluator**
   - Compares the response against an expected answer
   - Returns a similarity score between 0.0 (completely different) and 1.0 (identical)

3. **WordCountEvaluator**
   - Validates response length against word count constraints
   - Configurable minimum, maximum, or target word count

## Sample Evaluations

The `SampleEvals` directory contains example configurations:

- `engineer.yaml`: Evaluates engineering candidate summaries
- `consultant.yaml`: Evaluates consulting-style responses

## Development

### Adding New Evaluators

1. Create a new class in `evaluator.py` that implements the evaluation logic
2. Update the `allowed_evaluators()` method in `EvaluationRunner`
3. Add your evaluator to the YAML configuration file

### Running Tests

```bash
# Run the test suite
uv run pytest tests/
```

## License

[Specify License]

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- Built with Python's asyncio for efficient async operations
- Uses Pydantic for data validation and settings management
