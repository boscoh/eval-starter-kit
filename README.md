# AI Eval Starter Kit

A lightweight kit for quickly standing up LLM evals. It includes a simple web UI, a CLI, and a pluggable API layer that works with local models (Ollama) and major cloud providers (OpenAI, AWS Bedrock). Use it to compare prompts, models, and configurations, and to capture reproducible results you can share.

## Quick Start

### 1. Install

Clone the repository
```bash
git clone <repository-url>
cd eval-starter-kit
```

Install uv if not already installed
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Install dependencies and create virtual environment
```bash
uv sync
```

### 2. Configure Your AI Service 

1. Ollama (Local Models)

   [Ollama](https://ollama.ai/) installed and running
   ```bash
   # Start Ollama server (if not already running)
   ollama pull llama3.2
   ollama serve
   ```

   Models:

   - llama3.2, llama3.1, llama3
   - qwen2.5, qwen2
   - mistral, mixtral
   - Any model available through Ollama

2. OpenAI

   `OPENAI_API_KEY` set in your environment or `.env` file  
   
   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

   Models:

   - gpt-4, gpt-4-turbo
   - gpt-3.5-turbo
   - o1-preview, o1-mini

3. AWS Bedrock

   AWS credentials configured (via AWS CLI, profile, or environment variables)

   ```bash
   # Configure AWS credentials (choose one method):
   
   # Method 1: AWS CLI (recommended)
   aws configure
   
   # Method 2: Environment variables
   export AWS_ACCESS_KEY_ID=your-access-key
   export AWS_SECRET_ACCESS_KEY=your-secret-key
   export AWS_DEFAULT_REGION=us-east-1
   
   # Method 3: Use specific AWS profile
   export AWS_PROFILE=your-profile-name
   ```

   Note: Bedrock implementation uses the Converse API and is configured for Claude models with tool-calling support:

   - anthropic.claude-3-sonnet-20240229-v1:0
   - anthropic.claude-3-5-sonnet-20240620-v1:0

### 3. Run the Web Interface

1. Start the web server:
   ```bash
   uv run server.py
   ```

2. Open your browser to http://localhost:8000

### 4. Running Evaluations via CLI

This will bulk run all configs in `runs`:

```bash
uv run runner.py
```

## Run Configuration

Evaluation configurations are defined in YAML files in the `runs/` directory.

Example `runs/consultant.yaml`:

```yaml
name: "Engineering Candidate Evaluation"
query_ref: consultant
prompt_ref: candidate-summary
service: ollama  # or "openai" or "bedrock"
model: llama3.2
repeat: 2
evaluators:
- word_count
- coherence
- equivalence
```

Evaluators:
- `coherence`
   - Evaluates the logical flow and coherence of the response
   - Returns a score between 0.0 (incoherent) and 1.0 (highly coherent)
- `word_count`
   - Validates response length against word count constraints
   - Configurable minimum and maximum word counts
- `equivalence`
   - Evaluates the semantic equivalence of the response
   - Configurable minimum and maximum semantic similarity scores
- to extend the base `Evaluator` class to create custom evaluation metrics:
  - Create a new class in `evaluator.py` that implements the evaluation logic
  - Update the `allowed_evaluators()` method in `EvaluationRunner`
  - Add your evaluator to the YAML configuration file

## Project Structure

```
.
├── README.md
├── prompts/               # Reusable system prompts
├── queries/               # Prompt/query YAMLs used by runs
├── results/               # Example evaluation results
├── runs/                  # Run configurations (what to evaluate)
├── chat_client.py         # Client for interacting with the API
├── evaluator.py           # Core evaluation logic
├── index.html             # Simple web UI for the server
├── runner.py              # CLI for running evaluations
├── schemas.py             # Pydantic models
├── server.py              # Web server and API
├── setup_logger.py        # Logging configuration
├── yaml_utils.py          # YAML utility functions
├── test_chat.py           # Basic API/chat tests
├── test_embed.py          # Embedding-related tests
├── pyproject.toml         # Project metadata and dependencies
└── uv.lock                # Locked dependency versions for uv
```


