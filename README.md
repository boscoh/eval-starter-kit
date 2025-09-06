# AI Eval Starter Kit

A ready-to-use toolkit for setting up and running evaluations of AI model responses. This starter kit provides a web interface and API for assessing and comparing AI model outputs with minimal setup.

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
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer
- One of the following:
  - For local models: [Ollama](https://ollama.ai/) installed and running
  - For OpenAI: `OPENAI_API_KEY` set in your environment or `.env` file

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd eval-starter-kit
   ```

2. Install dependencies using uv:
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install dependencies and .venv
   uv sync
   ```

### Running the Web Interface

1. Start the web server:
   ```bash
   uv run server.py
   ```

2. Open your browser to http://localhost:8000

### Running Evaluations via CLI

1. **Using Ollama (Local Models)**
   ```bash
   # Start Ollama server (if not already running)
   ollama serve
   
   # Run an evaluation
   uv run runner.py jobs/engineer.yaml
   ```

2. **Using OpenAI**
   ```bash
   # Set your OpenAI API key
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   
   # Run an evaluation with OpenAI
   uv run runner.py jobs/engineer.yaml --service openai --model gpt-4
   ```

## Configuration

Evaluation configurations are defined in YAML files in the `jobs/` directory.

### Example Configuration

```yaml
# jobs/engineer.yaml
name: "Engineering Candidate Evaluation"
prompt: |
  You are an experienced engineering manager evaluating a candidate's technical skills.
  Analyze the following resume and provide a detailed assessment.

prompt: |
  Candidate Name: John Doe
  Position: Senior Software Engineer
  
  Experience:
  - 5 years at TechCorp as Full Stack Developer
  - Worked with Python, JavaScript, and AWS
  - Led team of 4 developers
  
  Skills: Python, React, AWS, Docker, CI/CD

model: "llama3.2"  # or "gpt-4", etc.
service: "ollama"  # or "openai"
temperature: 0.7
max_tokens: 1000

evaluators:
  - type: coherence
  - type: word_count
    min_words: 200
    max_words: 500
```

## Available Evaluators

1. **Coherence**
   - Evaluates the logical flow and coherence of the response
   - Returns a score between 0.0 (incoherent) and 1.0 (highly coherent)

2. **Word Count**
   - Validates response length against word count constraints
   - Configurable minimum and maximum word counts

3. **Custom Evaluators**
   - Extend the base `Evaluator` class to create custom evaluation metrics

## Project Structure

```
.
├── jobs/                  # YAML job configurations
├── results/               # Evaluation results
├── prompts/        # Reusable system prompts
├── chat_client.py         # Client for interacting with the API
├── evaluator.py           # Core evaluation logic
├── runner.py              # CLI for running evaluations
├── schemas.py             # Pydantic models
├── server.py              # Web server and API
└── util.py                # Utility functions
```

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

MIT

## Sample Evals

The `SampleEvals` directory contains example configurations:

- `engineer.yaml`: Evaluates engineering candidate summaries
- `consultant.yaml`: Evaluates consulting-style responses

## Development

### Adding New Evaluators

1. Create a new class in `evaluator.py` that implements the evaluation logic
2. Update the `allowed_evaluators()` method in `EvaluationRunner`
3. Add your evaluator to the YAML configuration file

## License

MIT

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

