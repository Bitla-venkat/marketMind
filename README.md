# MarketMind

**MarketMind** is an AI-powered multi-agent marketing platform for Indian small and medium-sized businesses (SMBs).

The project is being developed as a research-oriented system for **dynamic multi-agent orchestration**, where an orchestrator can dynamically select agents, construct execution plans, execute tasks, validate results, and re-plan when required.

---

## Current Development Pipeline

The current prototype is being built incrementally:

```text
Startup Description
        │
        ▼
   ICP Agent
        │
        ▼
Lead Discovery Agent
        │
        ▼
 Enrichment Agent
        │
        ▼
  Scoring Agent
        │
        ▼
 Research Agent
        │
        ▼
 Outreach Agent
```

The final system will replace the fixed pipeline with a dynamic orchestration layer.

---

# Development Environment

## Requirements

Recommended environment:

* **Python:** 3.12+
* **Git**
* **pip**
* **Virtual environment:** Python `venv`
* **OS:** Linux/macOS/Windows
* **Gemini API key**
* **Tavily API key**

The project currently uses:

* Google Gemini for LLM-based reasoning
* Tavily for web search
* Pydantic for structured data
* FastAPI components where required

---

# 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd finalyear_proj
```

Replace `<YOUR_REPOSITORY_URL>` with the Git repository URL.

---

# 2. Create a Virtual Environment

Using Python 3.12:

```bash
python3.12 -m venv .venv
```

Activate it.

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

After activation, your terminal should show something similar to:

```text
(.venv) user@computer:~/finalyear_proj$
```

---

# 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

# 4. Install Dependencies

If `requirements.txt` exists:

```bash
pip install -r requirements.txt
```

If dependencies have not been added to `requirements.txt yet, install the current core dependencies:

```bash
pip install google-genai python-dotenv tavily-python pydantic
```

If the project uses FastAPI:

```bash
pip install fastapi uvicorn
```

---

# 5. Configure API Keys

Create a `.env` file in the project root:

```bash
touch .env
```

Add:

```env
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Do **not** commit `.env` to Git.

The repository's `.gitignore` already excludes `.env`.

---

# 6. Verify the Environment

Check Python:

```bash
python --version
```

Expected:

```text
Python 3.12.x
```

Check that the packages are available:

```bash
python -c "import google.genai; import tavily; import pydantic; print('Environment OK')"
```

Expected:

```text
Environment OK
```

---

# 7. Project Structure

The current project structure is approximately:

```text
finalyear_proj/
│
├── agents/
│   ├── base.py
│   ├── icp_agent.py
│   ├── lead_discovery_agent.py
│   └── enrichment_agent.py
│
├── core/
│   ├── models.py
│   └── state.py
│
├── tools/
│   └── search_tool.py
│
├── utils/
│   └── llm_retry.py
│
├── main.py
├── .env
├── .gitignore
└── README.md
```

The orchestration layer will be added as development progresses.

---

# 8. Run the Current Prototype

Make sure the virtual environment is active:

```bash
source .venv/bin/activate
```

Then run:

```bash
python main.py
```

The current pipeline should perform:

```text
ICP Generation
      ↓
Lead Discovery
      ↓
Lead Enrichment
```

Example output:

```text
===== ICP =====

{
    ...
}

Searching: Retail companies India
Searching: Retail startups India

===== LEADS =====

{
    ...
}

===== ENRICHED LEADS =====

{
    ...
}
```

---

# 9. API Key Troubleshooting

## Gemini API error

If you see:

```text
GEMINI_API_KEY is not set
```

check that `.env` exists in the project root:

```bash
ls -la
```

and contains:

```env
GEMINI_API_KEY=...
```

The application calls:

```python
load_dotenv()
```

to load the environment variables.

---

## Tavily API error

Check:

```env
TAVILY_API_KEY=...
```

in `.env`.

Also verify that the search tool is correctly configured.

---

# 10. Gemini Model Fallback

The project uses a fallback mechanism for Gemini models.

The general flow is:

```text
Primary Model
      ↓
Failure?
      ↓
Fallback Model
      ↓
Failure?
      ↓
Next Fallback Model
```

This prevents the entire pipeline from failing when a model is temporarily unavailable.

For example:

```text
gemini-3.7-flash
        ↓
gemini-3.5-flash-lite
```

The exact available models may change over time.

---

# 11. Development Workflow

Before starting development:

```bash
git pull
source .venv/bin/activate
```

Create a feature branch:

```bash
git checkout -b feature/<feature-name>
```

Example:

```bash
git checkout -b feature/scoring-agent
```

After making changes:

```bash
git status
git add .
git commit -m "Add lead scoring agent"
```

Push the branch:

```bash
git push -u origin feature/scoring-agent
```

---

# 12. Important Security Rules

Never commit:

```text
.env
API keys
OAuth credentials
private tokens
passwords
service-account credentials
```

Before pushing code, check:

```bash
git status
```

If `.env` appears under files to be committed, **do not commit it**.

You can also check tracked files:

```bash
git ls-files
```

---

# 13. Research Development Roadmap

The system is being developed in stages.

### Phase 1 — Agent Layer

```text
ICP Agent
Lead Discovery Agent
Enrichment Agent
Scoring Agent
Research Agent
Outreach Agent
```

### Phase 2 — Static Baseline

Create a fixed pipeline:

```text
ICP
 ↓
Discovery
 ↓
Enrichment
 ↓
Scoring
 ↓
Research
 ↓
Outreach
```

This becomes the baseline for experiments.

### Phase 3 — Agent Registry

Create a registry containing:

```text
Agent
Capabilities
Cost
Expected latency
Reliability
Input requirements
Output types
```

### Phase 4 — Planner

Convert a user goal into a task graph:

```text
User Goal
    ↓
Planner
    ↓
Task DAG
```

### Phase 5 — Dynamic Executor

The executor determines:

* Which agent performs each task
* Task dependencies
* Parallel execution
* Conditional execution

### Phase 6 — Runtime Validation

Validate agent outputs before continuing.

```text
Agent
 ↓
Output
 ↓
Validator
 ↓
Valid?
 ├── Yes → Continue
 └── No  → Replan
```

### Phase 7 — Runtime Replanning

When new information, failures, or missing data occur:

```text
Current State
      ↓
Problem detected
      ↓
Replanner
      ↓
New Task Graph
      ↓
Continue execution
```

### Phase 8 — Optimization

The orchestrator will consider:

```text
Accuracy
Cost
Latency
Reliability
Task complexity
Agent capability
```

The goal is to dynamically select an appropriate execution strategy.

---

# 14. Research Evaluation

The final system will be evaluated against the fixed multi-agent baseline.

Key metrics include:

| Metric             | Description                                             |
| ------------------ | ------------------------------------------------------- |
| Task Success Rate  | Percentage of successfully completed tasks              |
| Lead Quality       | Quality/relevance of discovered leads                   |
| Latency            | Total execution time                                    |
| Cost               | LLM/API usage cost                                      |
| Agent Calls        | Number of agents invoked                                |
| Unnecessary Calls  | Agents invoked without contributing to the final result |
| Recovery Rate      | Successful recovery from agent/tool failures            |
| Planning Accuracy  | Quality of generated task plans                         |
| Replanning Success | Ability to recover through dynamic replanning           |

The central research question is:

> **Can dynamic multi-agent orchestration reduce execution cost and latency while maintaining or improving task and lead quality compared with a fixed multi-agent pipeline?**

---

# 15. Development Principles

### Keep agents independent

Each agent should have a clear responsibility.

```text
ICP Agent          → Define target customer
Discovery Agent    → Find potential companies
Enrichment Agent   → Gather company information
Scoring Agent      → Evaluate leads
Research Agent     → Perform deeper research
Outreach Agent     → Generate personalized outreach
```

### Keep orchestration separate

Agents should perform tasks.

The orchestrator should decide:

```text
What needs to be done?
Who should do it?
When should it run?
Can tasks run in parallel?
Is the result valid?
Should the plan change?
```

### Avoid framework dependency initially

The core orchestration mechanisms should first be implemented directly so that the research contribution is clear.

Frameworks such as LangGraph, CrewAI, or AutoGen can later be evaluated as comparison/reference implementations if required.

---

# 16. Current Status

```text
Development Status

[✓] Project environment
[✓] Gemini integration
[✓] Tavily search integration
[✓] Shared agent state
[✓] Base agent abstraction
[✓] ICP Agent
[✓] Lead Discovery Agent
[~] Enrichment Agent
[ ] Lead Scoring Agent
[ ] Research Agent
[ ] Outreach Agent
[ ] Static baseline
[ ] Agent Registry
[ ] Task representation
[ ] Planner
[ ] Dynamic Executor
[ ] Validator
[ ] Runtime Replanner
[ ] Cost/Latency policy
[ ] Experimental evaluation
```

---

## Quick Start

For an alrea
