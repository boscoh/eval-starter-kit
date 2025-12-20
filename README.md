# microeval

A lightweight evaluation framework for LLM testing. Supports local models (Ollama) and cloud providers (OpenAI, AWS Bedrock, Groq). Run evaluations via CLI or web UI, compare models and prompts, and track results.

## Quick Start

### 1. Configure API Keys

Create a `.env` file with your API keys:

```bash
# OpenAI
OPENAI_API_KEY=your-api-key-here

# Groq
GROQ_API_KEY=your-api-key-here

# AWS Bedrock (option 1: use a profile)
AWS_PROFILE=your-profile-name

# AWS Bedrock (option 2: use credentials directly)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1
```

For local models, install [Ollama](https://ollama.ai/) and run:
```bash
ollama pull llama3.2
ollama serve
```

### 2. Run a Demo

```bash
uv run microeval demo1
```

This creates a `summary-evals` directory with example evaluations and opens the web UI at http://localhost:8000.

Or try the JSON evaluation demo:
```bash
uv run microeval demo2
```

This creates a `json-evals` directory with structured output evaluations.

---

## Tutorial: Building Your First Evaluation

### Step 1: Create Your Evaluation Directory

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

Create `my-evals/prompts/summarizer.txt`:

```
You are a helpful assistant that summarizes text concisely.

## Instructions
- Summarize the given text in 2-3 sentences
- Capture the key points and main ideas
- Use clear, simple language

## Output Format
Return only the summary, no preamble or explanation.
```

The filename (without extension) becomes the `prompt_ref`.

### Step 3: Create a Query (Test Case)

Create `my-evals/queries/pangram.yaml`:

```yaml
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
```

- `input` - The text sent to the LLM (user message)
- `output` - The expected/ideal response (used by evaluators like `equivalence`)

The filename (without extension) becomes the `query_ref`.

### Step 4: Create a Run Configuration

Create `my-evals/runs/summarize-gpt4o.yaml`:

```yaml
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
```

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

**Web UI:**
```bash
uv run microeval ui my-evals
```

Navigate to http://localhost:8000, go to the **Runs** tab, and click the run button.

**CLI:**
```bash
uv run microeval run my-evals
```

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

Use the **Graph** tab in the Web UI to visualize and compare results across different runs.

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
min_words: 50
max_words: 200
target_words: 100
```

### Creating Custom Evaluators

1. Create a class in `microeval/evaluator.py`:

```python
class MyCustomEvaluator:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config

    async def evaluate(self, response_text: str) -> Dict[str, Any]:
        score = 1.0
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
    "mycustom": MyCustomEvaluator(run_config),
}
```

3. Update `EvaluationRunner.evaluators()` to include your evaluator name.

4. Use in your run config:
```yaml
evaluators:
- coherence
- mycustom
```

---

## Comparing Models and Prompts

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
uv run microeval run my-evals
```

Compare results in the Graph view.

### Compare Multiple Prompts

Create different prompts and run configs:

```
my-evals/prompts/
├── summarizer-basic.txt
├── summarizer-detailed.txt
└── summarizer-expert.txt

my-evals/runs/
├── test-basic.yaml           # prompt_ref: summarizer-basic
├── test-detailed.yaml        # prompt_ref: summarizer-detailed
└── test-expert.yaml          # prompt_ref: summarizer-expert
```

---

## CLI Commands

```bash
microeval                             # Show help
microeval ui BASE_DIR                 # Start web UI for evals directory
microeval run BASE_DIR                # Run all evaluations in directory
microeval demo1                       # Create summary-evals and launch UI
microeval demo2                       # Create json-evals and launch UI  
microeval chat SERVICE                # Interactive chat with LLM provider
```

### ui - Web Interface

```bash
microeval ui my-evals                 # Start UI on default port 8000
microeval ui my-evals --port 3000     # Use custom port
microeval ui my-evals --reload        # Enable auto-reload for development
```

### run - CLI Evaluation Runner

```bash
microeval run my-evals                # Run all configs in my-evals/runs/*.yaml
```

Runs all evaluation configs and saves results to `my-evals/results/`.

### demo1 / demo2 - Quick Start Demos

```bash
microeval demo1                       # Summary evaluation demo
microeval demo1 --base-dir custom     # Use custom directory name
microeval demo1 --port 3000           # Use custom port

microeval demo2                       # JSON/structured output demo
```

### chat - Interactive Chat

Test LLM providers directly:
```bash
microeval chat openai
microeval chat ollama
microeval chat bedrock
microeval chat groq
```

---

## Project Structure

```
.
├── .env                             # API keys
├── microeval/
│   ├── cli.py                       # CLI entry point
│   ├── server.py                    # Web server and API
│   ├── runner.py                    # Evaluation runner
│   ├── evaluator.py                 # Evaluation logic
│   ├── chat_client.py               # LLM provider clients
│   ├── chat.py                      # Interactive chat
│   ├── schemas.py                   # Pydantic models
│   ├── index.html                   # Web UI
│   ├── graph.py                     # Metrics visualization
│   ├── yaml_utils.py                # YAML helpers
│   ├── summary-evals/               # Demo 1: summary evaluations
│   └── json-evals/                  # Demo 2: JSON/structured output
└── my-evals/                        # Your evaluation project
    ├── prompts/
    ├── queries/
    ├── runs/
    └── results/
```

## Services and Models

Default models configured in `microeval/config.json`:

| Service  | Default Model              |
|----------|----------------------------|
| openai   | gpt-4o                     |
| bedrock  | amazon.nova-pro-v1:0       |
| ollama   | llama3.2                   |
| groq     | llama-3.3-70b-versatile    |

---

## Tips

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
