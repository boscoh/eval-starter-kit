# tinyeval

A lightweight evaluation framework for LLM testing. Supports local models (Ollama) and cloud providers (OpenAI, AWS Bedrock, Groq). Run evaluations via CLI or web UI, compare models and prompts, and track results.

## Quick Start

### 1. Install

Clone the repository:
```bash
git clone https://github.com/boscoh/eval-starter-kit
cd tinyeval
```

Install uv if not already installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies:
```bash
uv sync
```

### 2. Configure AI Service

**Ollama (Local Models)**

[Ollama](https://ollama.ai/) installed and running:
```bash
ollama pull llama3.2
ollama serve
```

Models: llama3.2, llama3.1, qwen2.5, mistral, mixtral, etc.

**OpenAI**

Set `OPENAI_API_KEY`:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

**AWS Bedrock**

Configure AWS credentials:
```bash
aws configure
# or set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
```

**Groq**

Set `GROQ_API_KEY`:
```bash
echo "GROQ_API_KEY=your-api-key-here" > .env
```

### 3. Run Web UI

Start the web UI (consultant evaluations by default):
```bash
uv run tinyeval ui
```

Or with engineer evaluations:
```bash
uv run tinyeval ui evals-engineer
```

Open http://localhost:8000

Tabs:
- **Runs** - Create and manage evaluation configurations
- **Queries** - Define test cases
- **Prompts** - Edit system prompts
- **Graph** - Visualize performance metrics

### 4. Run Evaluations via CLI

The project has two evaluation directories:
- `evals-consultant/` - Consultant candidate evaluations (default)
- `evals-engineer/` - Engineer candidate evaluations

**Run all configs:**
```bash
uv run tinyeval run evals-consultant
```

**Run with engineer directory:**
```bash
uv run tinyeval run evals-engineer
```

### 5. Interactive Chat

Test LLM providers interactively:
```bash
uv run tinyeval chat openai
uv run tinyeval chat bedrock
uv run tinyeval chat ollama
uv run tinyeval chat groq
```

## Run Configuration

Configurations are YAML files in `evals-*/runs/`. Example:

```yaml
name: "Candidate Evaluation"
query_ref: consultant
prompt_ref: candidate-summary
service: ollama  # or "openai", "bedrock", "groq"
model: llama3.2
repeat: 2
evaluators:
- word_count
- coherence
- equivalence
```

### Evaluators

- **coherence** - Logical flow and coherence (0.0-1.0)
- **word_count** - Response length validation (configurable min/max)
- **equivalence** - Semantic similarity (configurable min/max)

### Custom Evaluators

1. Create class in `tinyeval/evaluator.py` implementing `async evaluate(self, response_text: str) -> Dict[str, Any]`
2. Register in `EvaluationRunner.__init__` and `EvaluationRunner.evaluators()`
3. Add (lowercase) to YAML config under `evaluators`

## Project Structure

```
.
├── README.md
├── pyproject.toml
├── .env                             # API keys (create from .env.example)
├── .env.example
├── tinyeval/                        # Main package
│   ├── __init__.py
│   ├── cli.py                       # CLI entry point (ui, run, chat)
│   ├── server.py                    # Web server and API
│   ├── runner.py                    # Evaluation runner
│   ├── evaluator.py                 # Evaluation logic
│   ├── chat_client.py               # LLM provider clients
│   ├── chat.py                      # Interactive chat
│   ├── schemas.py                   # Pydantic models
│   ├── config.json                  # Model configuration
│   ├── index.html                   # Web UI
│   ├── graph.py                     # Metrics visualization
│   ├── setup_logger.py              # Logging config
│   └── yaml_utils.py                # YAML helpers
├── evals-consultant/                # Consultant evaluation configs (default)
│   ├── prompts/
│   ├── queries/
│   ├── runs/
│   └── results/
├── evals-engineer/                  # Engineer evaluation configs
│   ├── prompts/
│   ├── queries/
│   ├── runs/
│   └── results/
└── uv.lock                          # Locked dependencies
```

## CLI Commands

```bash
uv run tinyeval ui [EVALS_DIR]       # Start web UI
uv run tinyeval run EVALS_DIR        # Run all evaluations
uv run tinyeval chat SERVICE         # Interactive chat (openai, bedrock, ollama, groq)
```

## Services and Models

Default models configured in `tinyeval/config.json`:

- **openai**: `gpt-4o`
- **bedrock**: `amazon.nova-pro-v1:0`
- **ollama**: `llama3.2`
- **groq**: `llama-3.3-70b-versatile`
