# 1. Lab Title

## Context Quarantine: Multi-Agent Systems with LangGraph Supervisor

# What is Context Engineering?

**Context Engineering** is the art and science of filling the context window with just the right information at each step of an agent’s trajectory.

Unlike basic prompt engineering, which focuses mainly on instructions, Context Engineering focuses on **information architecture**. As LLM context windows grow larger (up to millions of tokens), simply stuffing them with all available data often leads to performance degradation. Problems like _Context Distraction_ (where the model loses focus on the core task) or _Context Clash_ (where mixed, unrelated information confuses the model's reasoning) become common.

Context Engineering involves deliberate techniques and strategies like **Context Quarantine** (isolating specific sub-tasks into separate, pristine context windows) and **Context Pruning** (removing noise) to ensure the LLM receives exactly what it needs, when it needs it, yielding more accurate and reliable results in complex agentic systems.

# 2. Problem Statement / Use Case Overview

How do we design systems that maintain only the most relevant, structured, and useful context for an agent—while discarding or compressing everything else?

One of the most popular and intuitive ways to isolate context (**Context Quarantine**) is to split work across sub-agents. A motivation for the OpenAI [Swarm](https://github.com/openai/swarm) library was “[separation of concerns](https://openai.github.io/openai-agents-python/ref/agent/)”, where a team of agents can handle sub-tasks. Each agent has a specific set of tools, instructions, and its own context window.

In LangGraph's multi-agent system, a popular and intuitive way to implement **Context Quarantine** is the [supervisor](https://github.com/langchain-ai/langgraph-supervisor-py) architecture, which is the same pattern used in Anthropic's [multi-agent researcher](https://www.anthropic.com/engineering/built-multi-agent-research-system). This allows the supervisor to delegate tasks to sub-agents, each with their own context window.

# 3. Input Data

| Item                    | Detail                                                                                                                            |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| User query              | Natural-language question requiring research + math (e.g., _"What is the combined headcount of the MAANG companies in 2024?"_)    |
| FAANG headcount data    | Mock static dataset returned by `web_search()` (2024 figures)                                                                     |
| AWS Bedrock Credentials | Access Key ID, Secret Access Key, Endpoint URL, and Region — required to authenticate requests to the Claude model on AWS Bedrock |

# 4. Processing

```
User Query
    │
    ▼
Supervisor Agent  ──── analyse task ────►  Research Agent  (web_search tool)
    │                                              │
    │                                        facts / data
    │◄──────────────────────────────────────────────
    │
    ├──── if math needed ───►  Math Agent  (add / multiply tools)
    │                                  │
    │◄────────── result ────────────────
    │
    ▼
Final synthesised answer returned to the user
```

1. The **Supervisor** receives the user's query and decides which specialist(s) to call.
2. The **Research Agent** fetches company headcount data using `web_search`.
3. The **Math Agent** sums the retrieved numbers using `add` / `multiply`.
4. The Supervisor collects results and produces the final answer.

# 5. Output

A final natural-language answer synthesised from the two specialist agents, e.g.:

> _"The combined headcount of the MAANG companies in 2024 is approximately 1,977,586 employees."_

Additionally, a **Mermaid architecture diagram** of the compiled LangGraph workflow is rendered inline in the notebook.

# 6. Tech Stack

| Layer                     | Technology                                                                               |
| ------------------------- | ---------------------------------------------------------------------------------------- |
| LLM                       | Anthropic Claude Sonnet 4.5 on AWS Bedrock (`global.anthropic.claude-sonnet-4-5-20250929-v1:0`) |
| Agent Framework           | LangChain + LangGraph                                                                    |
| Multi-Agent Orchestration | `langgraph_supervisor`                                                                   |
| Agent Type                | ReAct (`create_react_agent` from `langgraph.prebuilt`)                                   |
| Language                  | Python 3.11                                                                              |
| Runtime                   | Google Colab / Jupyter Notebook                                                          |

# 7. Underlying Concepts

- **Context Quarantine** — Isolating sub-tasks in separate context windows to prevent context clash and context distraction.
- **Context Clash** — Conflicting information accumulating in a single context window degrades model output quality.
- **Context Distraction** — A context window growing so long that the model over-weights it relative to its own training knowledge.
- **Supervisor Architecture** — A hierarchical multi-agent pattern where a central supervisor routes tasks to specialized sub-agents.
- **ReAct Agent** — An agent that alternates between _Reasoning_ (thinking) and _Acting_ (using tools), iterating until it reaches an answer.
- **Tool-based Handoff** — Agents communicate through specialized handoff tools, enabling clean context boundaries.

> Refer to Study Note: **Context Engineering — Context Quarantine & Multi-Agent Systems**

# 8. Pre-requisites

- Basic familiarity with Python (functions, classes, `import` statements).
- **AWS Bedrock credentials** (Access Key ID, Secret Access Key, Endpoint URL, and Region) — provided in the lab environment. Click the 🔑 key icon in the top-right corner of the lab to obtain your credentials.
- High-level understanding of what an LLM is and what a "context window" means.
- (Optional) Basic awareness of LangChain agents and tools.

# 9. Environment / Dependencies Setup

The cell below installs all required Python packages:

| Package                | Purpose                                                     |
| ---------------------- | ----------------------------------------------------------- |
| `langchain`            | Core LLM abstraction and agent utilities                    |
| `langgraph`            | Stateful, graph-based agent workflow engine                 |
| `langgraph_supervisor` | Prebuilt supervisor multi-agent pattern for LangGraph       |
| `langchain-aws`        | LangChain integration for AWS Bedrock models                |
| `boto3`                | AWS SDK for Python — used to connect to the Bedrock runtime |

Run this cell first — it only needs to be run once per session.

```python
!pip install -q langchain langgraph langgraph_supervisor langchain-aws boto3
```

### Set up your AWS Bedrock Credentials

To communicate with the Claude model on AWS Bedrock, you need your AWS credentials. Click the 🔑 key icon in the **top-right corner** of the lab environment to obtain the following:

- **Access Key ID**
- **Secret Access Key**
- **Endpoint URL**
- **Region**

Paste them into the cell below. The cell creates a `boto3` Bedrock Runtime client that LangChain will use for all LLM calls.

```python
import boto3

# AWS Bedrock credentials — paste your own values from the lab's key icon
ACCESS_KEY_ID = "Your Access key ID"
SECRET_ACCESS_KEY = "Your Secret Access Key"
REGION = "Your Region"

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY
)

print("AWS Bedrock client created successfully.")

```

### Build the Multi-Agent System

This is the heart of the lab. Split the cell into four smaller steps so each code block matches one idea.

#### 1) Import libraries and initialize the LLM

```python
from langchain_aws import ChatBedrockConverse
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from IPython.display import Image, display

llm = ChatBedrockConverse(
    client=bedrock_client,
    model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    temperature=0,
)
```

#### 2) Define the tools

```python
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def web_search(query: str) -> str:
    """Mock web search function that returns FAANG company headcounts."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Meta**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )
```

#### 3) Create the specialist agents

```python
math_agent = create_react_agent(
    model=llm,
    tools=[add, multiply],
    name="math_expert",
    prompt="""You are a specialized mathematics expert with access to addition and multiplication tools.

Your responsibilities:
- Solve mathematical problems using the available tools
- Always use tools for calculations rather than computing mentally
- Use one tool at a time and show your work clearly
- Focus exclusively on mathematical computations

Constraints:
- Do NOT attempt research, web searches, or data gathering
- Do NOT perform calculations without using the provided tools
- Always explain your mathematical reasoning step by step"""
)

research_agent = create_react_agent(
    model=llm,
    tools=[web_search],
    name="research_expert",
    prompt="""You are a specialized research expert with access to web search capabilities.

Your responsibilities:
- Find and retrieve factual information using web search
- Provide comprehensive, well-sourced answers to research questions
- Focus on data gathering and information synthesis

Constraints:
- Do NOT perform mathematical calculations or computations
- Do NOT attempt to solve math problems - delegate those to the math expert
- Always use your search tool to find current, accurate information
- Present findings clearly and cite sources when available"""
)
```

#### 4) Create the supervisor and compile the workflow

```python
supervisor_prompt = """You are an intelligent team supervisor managing two specialized experts: a research expert and a math expert.

Your role is to:
1. Analyze incoming requests to determine the required expertise
2. Delegate tasks to the appropriate specialist
3. Coordinate between agents when tasks require multiple skills
4. Synthesize results from multiple agents when necessary

Delegation Rules:
- For data gathering, company information, current events, or factual research -> use research_agent
- For calculations, mathematical operations, or numerical analysis -> use math_agent
- For complex tasks requiring both research and math -> delegate sequentially (research first, then math)

Important: You are a coordinator, not a doer. Always delegate work to your specialists rather than attempting tasks yourself. Never perform calculations or research directly."""

workflow = create_supervisor([research_agent, math_agent], model=llm, prompt=supervisor_prompt)
app = workflow.compile()
display(Image(app.get_graph(xray=True).draw_mermaid_png()))
```

### Run a Query Through the Multi-Agent System

Now let's put our system to the test. Split this cell into three parts so the output is easier to follow.

#### 1) Define the message formatter

```python
def format_messages(messages):
    for msg in messages:
        role = getattr(msg, "type", type(msg).__name__)
        content = getattr(msg, "content", str(msg))
        if content:
            print(f"[{role.upper()}]\n{content}\n{'─' * 60}")
```

#### 2) Send the query through the workflow

```python
query = "what's the combined headcount of the MAANG companies in 2024?"
result = app.invoke({"messages": [{"role": "user", "content": query}]})
format_messages(result["messages"])
```
#### 2) Send the query through the workflow