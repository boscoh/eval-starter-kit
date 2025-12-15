# starteval

A lightweight evaluation framework for LLM testing. Supports local models (Ollama) and cloud providers (OpenAI, AWS Bedrock, Groq). Run evaluations via CLI or web UI, compare models and prompts, and track results.

## Quick Start

### 1. Install

Clone the repository:
```bash
git clone https://github.com/boscoh/starteval
cd starteval
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

For local development, the easiest approach is to set up an AWS profile via `~/.aws/config` and then specify it with `AWS_PROFILE`:

```bash
aws configure --profile your-profile-name
# or manually edit ~/.aws/config

echo "AWS_PROFILE=your-profile-name" >> .env
```

Alternatively, you can set credentials directly:
```bash
echo "AWS_ACCESS_KEY_ID=your-access-key" >> .env
echo "AWS_SECRET_ACCESS_KEY=your-secret-key" >> .env
echo "AWS_DEFAULT_REGION=us-east-1" >> .env
```

Note: AWS profiles are recommended for local dev since key rotation is handled automatically.

**Groq**

Set `GROQ_API_KEY`:
```bash
echo "GROQ_API_KEY=your-api-key-here" > .env
```

### 3. Run Your First Evaluation

Try the sample evaluation:
```bash
uv run starteval ui sample-evals
```

Open http://localhost:8000 to see the web UI.

---

## Tutorial: Building Your First Evaluation

This tutorial walks you through creating an evaluation from scratch. We'll build a text summarization evaluator.

### Step 1: Create Your Evaluation Directory

Each evaluation project lives in its own directory with four subdirectories:

```bash
mkdir -p my-evals/{prompts,queries,runs,results}
```

This creates:
```
my-evals/
├── prompts/    # System prompts (instructions for the LLM)
├── queries/    # Test cases (input/output pairs)
├── runs/       # Run configurations (which model, prompt, query to use)
└── results/    # Generated results (created automatically)
```

### Step 2: Write a System Prompt

Create a prompt file that tells the LLM how to behave:

```bash
cat > my-evals/prompts/summarizer.txt << 'EOF'
You are a helpful assistant that summarizes text concisely.

## Instructions
- Summarize the given text in 2-3 sentences
- Capture the key points and main ideas
- Use clear, simple language

## Output Format
Return only the summary, no preamble or explanation.
EOF
```

Prompts are `.txt` files in the `prompts/` directory. The filename (without extension) becomes the `prompt_ref`.

### Step 3: Create a Query (Test Case)

Create a query file with input/output pairs for testing:

```bash
cat > my-evals/queries/pangram.yaml << 'EOF'
---
input: >-
  The quick brown fox jumps over the lazy dog. This sentence is famous
  because it contains every letter of the English alphabet at least once.
  It has been used for centuries to test typewriters, fonts, and keyboards.
  The phrase was first used in the late 1800s and remains popular today
  for testing purposes.
output: >-
  The sentence "The quick brown fox jumps over the lazy dog" is a pangram
  containing every letter of the alphabet. It has been used since the late
  1800s to test typewriters, fonts, and keyboards.
EOF
```

**Query structure:**
- `input` - The text sent to the LLM (user message)
- `output` - The expected/ideal response (used by evaluators like `equivalence`)

The filename (without extension) becomes the `query_ref`.

### Step 4: Create a Run Configuration

Create a run configuration that ties everything together:

```bash
cat > my-evals/runs/summarize-gpt4o.yaml << 'EOF'
---
query_ref: pangram
prompt_ref: summarizer
service: openai
model: gpt-4o
repeat: 3
temperature: 0.5
evaluators:
- word_count
- coherence
- equivalence
EOF
```

**Run configuration fields:**

| Field         | Description                                                  |
|---------------|--------------------------------------------------------------|
| `query_ref`   | Name of the query file (without `.yaml`)                     |
| `prompt_ref`  | Name of the prompt file (without `.txt`)                     |
| `service`     | LLM provider: `openai`, `bedrock`, `ollama`, or `groq`       |
| `model`       | Model name (e.g., `gpt-4o`, `llama3.2`)                      |
| `repeat`      | Number of times to run the evaluation                        |
| `temperature` | Sampling temperature (0.0 = deterministic)                   |
| `evaluators`  | List of evaluators to run                                    |

### Step 5: Run the Evaluation

**Option A: Web UI**
```bash
uv run starteval ui my-evals
```

Navigate to http://localhost:8000, go to the **Runs** tab, and click the run button.

**Option B: CLI**
```bash
uv run starteval run my-evals
```

This runs all configurations in `my-evals/runs/`.

### Step 6: View Results

Results are saved to `my-evals/results/` as YAML files:

```yaml
---
texts:
- "The sentence 'The quick brown fox...' is notable for..."
- "The phrase 'The quick brown fox...' contains every letter..."
- "The quick brown fox jumps over the lazy dog is a famous..."
evaluations:
- name: word_count
  values: [1.0, 1.0, 1.0]
  average: 1.0
  standard_deviation: 0.0
- name: coherence
  values: [0.95, 0.92, 0.98]
  average: 0.95
  standard_deviation: 0.03
- name: equivalence
  values: [0.88, 0.91, 0.85]
  average: 0.88
  standard_deviation: 0.03
```

**Result structure:**
- `texts` - All generated responses from each run
- `evaluations` - Scores from each evaluator with statistics

In the Web UI, use the **Graph** tab to visualize and compare results across different runs.

---

## Evaluators

Evaluators score responses on a 0.0-1.0 scale:

| Evaluator      | Description                       | How it Works                               |
|----------------|-----------------------------------|-------------------------------------------|
| `coherence`    | Logical flow and clarity          | LLM scores structure and consistency       |
| `equivalence`  | Semantic similarity to expected   | LLM compares meaning with query output     |
| `word_count`   | Response length validation        | Algorithmic check (no LLM call)            |

### Word Count Configuration

Add these optional fields to your run config:

```yaml
min_words: 50    # Minimum word count
max_words: 200   # Maximum word count
target_words: 100  # Target word count (scores based on distance)
```

### Creating Custom Evaluators

1. Create a class in `starteval/evaluator.py`:

```python
class MyCustomEvaluator:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config

    async def evaluate(self, response_text: str) -> Dict[str, Any]:
        # Your evaluation logic here
        score = 1.0  # Calculate your score (0.0 to 1.0)
        return {
            "score": score,
            "text": "Evaluation details",
            "elapsed_ms": 0,
            "token_count": 0,
        }
```

2. Register in `EvaluationRunner.__init__`:
```python
self.evaluators = {
    "coherence": CoherenceEvaluator(chat_client, run_config),
    "equivalence": EquivalenceEvaluator(chat_client, run_config),
    "word_count": WordCountEvaluator(run_config),
    "mycustom": MyCustomEvaluator(run_config),  # Add this
}
```

3. Update the static method `EvaluationRunner.evaluators()` to include your evaluator name.

4. Use in your run config:
```yaml
evaluators:
- coherence
- mycustom
```

---

## Comparing Models and Prompts

A key use case is comparing different models or prompts on the same test cases.

### Compare Multiple Models

Create multiple run configs with the same query and prompt but different models:

```
my-evals/runs/
├── summarize-gpt4o.yaml      # service: openai, model: gpt-4o
├── summarize-claude.yaml     # service: bedrock, model: anthropic.claude-3-sonnet
├── summarize-llama.yaml      # service: ollama, model: llama3.2
└── summarize-groq.yaml       # service: groq, model: llama-3.3-70b-versatile
```

Run all:
```bash
uv run starteval run my-evals
```

Compare results in the Graph view.

### Compare Multiple Prompts

Create different prompts and run configs:

```
my-evals/prompts/
├── summarizer-basic.txt      # Simple instructions
├── summarizer-detailed.txt   # Detailed step-by-step
└── summarizer-expert.txt     # Expert persona

my-evals/runs/
├── test-basic.yaml           # prompt_ref: summarizer-basic
├── test-detailed.yaml        # prompt_ref: summarizer-detailed
└── test-expert.yaml          # prompt_ref: summarizer-expert
```

---

## Web UI Guide

Start the UI:
```bash
uv run starteval ui my-evals
```

### Tabs

| Tab         | Purpose                                              |
|-------------|------------------------------------------------------|
| **Runs**    | Create, edit, and execute run configurations         |
| **Queries** | Define and edit test cases (input/output pairs)      |
| **Prompts** | Write and manage system prompts                      |
| **Graph**   | Visualize evaluation results and compare runs        |

### Workflow

1. **Prompts** → Write your system prompt
2. **Queries** → Define your test case
3. **Runs** → Configure which model, prompt, and query to use
4. **Runs** → Click the run button to execute
5. **Graph** → View and compare results

---

## CLI Commands

```bash
uv run starteval ui [EVALS_DIR]       # Start web UI (default: evals-consultant)
uv run starteval run EVALS_DIR        # Run all evaluations in directory
uv run starteval chat SERVICE         # Interactive chat (openai, bedrock, ollama, groq)
uv run starteval demo                 # Create sample-evals and run if not exists
```

### Demo

Create sample evaluations and launch the UI:
```bash
uv run starteval demo
# Creates sample-evals directory and opens the web UI at http://localhost:8000
```

The demo includes evaluations for all supported services (OpenAI, Bedrock, Ollama, Groq) using the same prompt and test case for easy comparison.

### Interactive Chat

Test LLM providers directly:
```bash
uv run starteval chat openai
uv run starteval chat ollama
uv run starteval chat bedrock
uv run starteval chat groq
```

---

## Project Structure

```
.
├── README.md
├── pyproject.toml
├── .env                             # API keys (create from .env.example)
├── starteval/                        # Main package
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
│   └── yaml_utils.py                # YAML helpers
├── sample-evals/                    # Example evaluation project
│   ├── prompts/
│   ├── queries/
│   ├── runs/
│   └── results/
└── uv.lock
```

## Services and Models

Default models configured in `starteval/config.json`:

| Service  | Default Model              |
|----------|----------------------------|
| openai   | gpt-4o                     |
| bedrock  | amazon.nova-pro-v1:0       |
| ollama   | llama3.2                   |
| groq     | llama-3.3-70b-versatile    |

---

## Tips and Best Practices

### Prompt Engineering
- Start with simple prompts and iterate
- Use clear section headers (## Instructions, ## Output Format)
- Specify output format explicitly
- Test with `temperature: 0.0` first for deterministic results

### Evaluation Design
- Use `repeat: 3` or higher to account for model variability
- Include `equivalence` when you have a known-good answer
- Use `coherence` for open-ended responses
- Create multiple query files to test different scenarios

### Comparing Results
- Keep one variable constant when comparing (e.g., same prompt, different models)
- Use the Graph tab to visualize trends
- Check standard deviation to understand consistency
