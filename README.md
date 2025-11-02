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

## Interactive Test Clients

This repository includes three interactive test clients that demonstrate progressively sophisticated patterns of LLM integration:

**Quick Guide - Which client should I use?**

- **Want to just chat with an LLM?** → Use `test_chat.py`
- **Testing embedding/RAG retrieval?** → Use `test_rag_chat.py`
- **Exploring agentic/tool-calling patterns?** → Use `test_mcp_chat.py`

All clients support environment-based configuration via `LLM_SERVICE` (set to `openai`, `bedrock`, or `ollama`).

---

### 1. Basic Chat Client (`test_chat.py`)

**Pure LLM interaction** - Simple chat loop for testing LLM providers without any RAG or tools.

**What it does:**
- Direct conversational interface with any LLM provider
- No data retrieval, no tools, no RAG - just pure chat
- Useful for quick model testing and comparing provider responses

**Configuration:**
- Default service: `openai`
- Configure via `LLM_SERVICE` environment variable
- Supports: `openai`, `bedrock`, `ollama`

**Usage:**
```bash
# Use default (OpenAI)
uv run test_chat.py

# Or specify service
env LLM_SERVICE=bedrock uv run test_chat.py
env LLM_SERVICE=ollama uv run test_chat.py
```

**Example interaction:**
```
Chat loop with openai-gpt-4o

You: Hello!
Response: Hello! How can I help you today?
```

---

### 2. RAG Chat Client (`test_rag_chat.py`)

**RAG + LLM synthesis** - Demonstrates the classic RAG pattern: retrieve relevant data via embeddings, then synthesize with LLM.

**What it does:**
1. Embeds your query using the embedding service
2. Finds the best matching speaker via cosine distance to all speaker embeddings
3. Calls the LLM to generate a natural language explanation of why the speaker matches
4. Returns formatted results with speaker details and AI explanation

**Key features:**
- Direct RAG implementation (no MCP server needed)
- Detailed logging of embedding vectors and distance calculations
- Shows distances to all speakers for debugging
- Separate distance metrics for bio and abstract embeddings
- Formatted markdown output with sections for speaker, bio, abstract, and explanation

**Configuration:**
- Default service: `openai`
- Configure via `LLM_SERVICE` environment variable
- Supports: `openai`, `bedrock`, `ollama`

**Usage:**
```bash
# Use default (OpenAI)
uv run test_rag_chat.py

# Or specify service
env LLM_SERVICE=bedrock uv run test_rag_chat.py
env LLM_SERVICE=ollama uv run test_rag_chat.py
```

**What you'll see in logs:**
- Query embedding: `embedding(1536)[0.123 0.456 ...]`
- Speaker distances: `0.234 0.567 0.890 ...`
- Best match details with separate bio and abstract distances
- Bio and abstract embedding representations

**Example output:**
```
# Best Match

## Speaker
Dr. Jane Smith

## Bio
Expert in distributed systems...

## Abstract
This talk explores...

# Why this speaker matches your query
Dr. Smith's expertise in distributed systems architecture 
directly addresses your query about scalability patterns...
```

---

### 3. MCP Tool-Augmented Chat Client (`test_mcp_chat.py`)

**Agentic workflow** - Advanced multi-step reasoning where the LLM autonomously uses tools to gather information before answering.

**What it does:**
1. Launches the MCP server automatically in the background
2. Fetches available tools from the server
3. Runs an agentic loop where the model:
   - Analyzes your query
   - Decides which tools to call and with what parameters
   - Calls tools (potentially multiple times across multiple rounds)
   - Synthesizes gathered information into a comprehensive answer
4. Supports up to 5 reasoning iterations per query

**Key features:**
- Autonomous multi-step reasoning and tool calling
- Prevents duplicate tool calls (tracks what's been called)
- Sophisticated prompting to encourage exploration and verification
- Detailed logging of each reasoning step and tool invocation
- Uses MCP protocol for tool discovery and execution

**Available tools:**
- `get_best_speaker(query: str)` - Find best matching speaker
- `list_all_speakers()` - Get all available speakers

**Configuration:**
- Default service: `openai`
- Configure via `LLM_SERVICE` environment variable
- Supports: `openai`, `bedrock` (requires tool-calling capable models)
- **Note:** Does NOT support `ollama` (most Ollama models lack tool-calling)

**Usage:**
```bash
# Use default (OpenAI GPT-4o)
uv run test_mcp_chat.py

# Or use Bedrock Claude
env LLM_SERVICE=bedrock uv run test_mcp_chat.py
```

**Example reasoning flow:**
```
Reasoning step 1 with 1 tool calls
Calling tool get_best_speaker({"query": "machine learning"})...
Tool result: [speaker details]

Reasoning step 2 with 1 tool calls
Calling tool list_all_speakers()...
Tool result: [all speakers]

Response: Based on my search, Dr. Jane Smith is the best match 
because...
```

---

### Comparison of Test Clients

| Feature                  | test_chat.py     | test_rag_chat.py         | test_mcp_chat.py         |
|--------------------------|------------------|--------------------------|--------------------------|
| **Pattern**              | Pure LLM         | RAG + LLM synthesis      | Agentic workflow         |
| **Complexity**           | Simple           | Moderate                 | Advanced                 |
| **Direct LLM chat**      | ✓                | ✓ (for explanation)      | ✓                        |
| **RAG/embeddings**       | ✗                | ✓ (direct)               | ✓ (via tools)            |
| **Tool calling**         | ✗                | ✗                        | ✓                        |
| **Multi-step reasoning** | ✗                | ✗                        | ✓ (up to 5 iterations)   |
| **MCP server**           | ✗                | ✗                        | ✓                        |
| **Output format**        | Plain text       | Markdown sections        | Plain text               |
| **Logging detail**       | Minimal          | Detailed (embeddings)    | Detailed (tool calls)    |
| **Ollama support**       | ✓                | ✓                        | ✗                        |
| **Supported services**   | All 3            | All 3                    | OpenAI, Bedrock only     |
| **Best for**             | Quick testing    | RAG debugging/synthesis  | Testing agentic patterns |
| **Demonstrates**         | Basic LLM usage  | Embedding + retrieval    | Autonomous reasoning     |

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
├── test_chat.py                 # Basic interactive chat client
├── test_rag_chat.py             # RAG-based speaker matching client
├── test_mcp_chat.py             # MCP tool-augmented chat client with multi-step reasoning
├── pyproject.toml               # Project metadata and dependencies
└── uv.lock                      # Locked dependency versions for uv
```
