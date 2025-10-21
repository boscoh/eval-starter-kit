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

   Models (examples):

   - llama3.2, llama3.1, llama3
   - qwen2.5, qwen2
   - mistral, mixtral
   - any model available through Ollama

2. OpenAI

   `OPENAI_API_KEY` set in your environment or `.env` file  
   
   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

   Models (examples):

   - gpt-4o
   - other chat/embedding models supported by your account

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

   Note: Bedrock implementation uses the Converse API and is configured for Claude models with tool-calling support (examples):

   - anthropic.claude-3-sonnet-20240229-v1:0
   - anthropic.claude-3-5-sonnet-20240620-v1:0

### 3. Run the Web Interface

1. Start the web server:
   ```bash
   uv run server.py
   ```

2. Open your browser to http://localhost:8000

### 4. Running Evaluations via CLI

- Run all configs in `runs/`:

```bash
uv run runner.py
```

- Run a specific config:

```bash
uv run runner.py runs/consultant.yaml
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

To add a custom evaluator:
- Create a new class in `evaluator.py` that implements `async evaluate(self, response_text: str) -> Dict[str, Any]`
- Register it in `EvaluationRunner.__init__` (add to `self.evaluators` mapping) and in `EvaluationRunner.evaluators()` (to expose it in the UI)
- Add your evaluator (lowercase key) to the YAML configuration file under `evaluators`

## Services and Models Defaults (as used by the Web UI)

- Services: `bedrock`, `ollama`, `openai`
- Example models per service:
  - bedrock: `anthropic.claude-3-sonnet-20240229-v1:0`
  - ollama: `llama3.2`
  - openai: `gpt-4o`

## RAG Utilities

This repo includes a simple RAG (Retrieval-Augmented Generation) helper used by the MCP server to recommend speakers.

- File: `rag.py`
- Data: `data/2025-09-02-speaker-bio.csv`
- Embeddings cache (auto-generated):
  - OpenAI: `data/embeddings-text-embedding-3-small.json`
  - Bedrock: `data/embeddings-amazon-titan-embed-text-v-2-0.json`

Service selection for embeddings is controlled by `LLM_SERVICE` env var (`openai`, `ollama`, or `bedrock`).

Generate or refresh embeddings (optional):
```bash
uv run rag.py
```

## MCP Server (Optional)

The repo includes an MCP server that exposes tools for speaker recommendation via the RAG utility.

- File: `mcp_server.py`
- Tools:
  - `get_best_speaker(query: str)`
  - `list_all_speakers()`
- Environment:
  - Set `LLM_SERVICE` to select the backend (`openai`, `ollama`, or `bedrock`)
  - For OpenAI, ensure `OPENAI_API_KEY` is set (e.g., in `.env`)

Run the MCP server:
```bash
uv run mcp_server.py
```

## MCP Chat Loop Client (test_mcp_chat.py)

This repository includes an interactive MCP chat loop client that connects to the MCP server over stdio and demonstrates tool-augmented reasoning to answer questions about conference speakers.

- File: `test_mcp_chat.py`
- What it does: Launches the MCP server automatically, fetches available tools, and runs a multi-step chat loop where the model can call tools, analyze results, and iterate until it has enough information to produce a final answer.
- Default backend: Bedrock Claude Sonnet via Converse API
- Supported backends: `bedrock`, `openai`

Run the chat loop (default Bedrock):
```bash
env LLM_SERVICE=bedrock uv run test_mcp_chat.py
```

Run with OpenAI instead:
```bash
env LLM_SERVICE=openai OPENAI_API_KEY=your-key uv run test_mcp_chat.py
```

Tips:
- The client prevents duplicate tool calls and supports up to 5 reasoning/tool-call iterations per question.
- It uses the same RAG-powered tools exposed by `mcp_server.py` (e.g., `get_best_speaker`, `list_all_speakers`).

## Project Structure

```
.
├── README.md
├── data/                        # Source data and cached embeddings
│   ├── 2025-09-02-speaker-bio.csv
│   ├── embeddings-amazon-titan-embed-text-v-2-0.json
│   └── embeddings-text-embedding-3-small.json
├── prompts/                     # Reusable system prompts
│   ├── candidate-skills.txt
│   └── candidate-summary.txt
├── queries/                     # Prompt/query YAMLs used by runs
│   ├── consultant.yaml
│   └── engineer.yaml
├── results/                     # Evaluation results (written by runner)
│   ├── consultant.yaml
│   └── engineer.yaml
├── runs/                        # Run configurations (what to evaluate)
│   ├── consultant.yaml
│   └── engineer.yaml
├── chat_client.py               # Client for interacting with the API providers
├── evaluator.py                 # Core evaluation logic and evaluators
├── index.html                   # Simple web UI for the server
├── mcp_server.py                # MCP server exposing RAG-based tools
├── rag.py                       # RAG utility (embeddings + retrieval)
├── runner.py                    # CLI for running evaluations
├── schemas.py                   # Pydantic models and schema helpers
├── server.py                    # Web server and API
├── setup_logger.py              # Logging configuration
├── yaml_utils.py                # YAML utility functions
├── test_chat.py                 # Basic API/chat tests
├── test_embed.py                # Embedding-related tests
├── test_mcp_chat.py             # MCP chat loop client (interactive)
├── pyproject.toml               # Project metadata and dependencies
└── uv.lock                      # Locked dependency versions for uv
```
