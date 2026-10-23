# CRS - Local AI Agent

A tool-using AI agent that runs on your machine. It uses Google Gemini for
reasoning and calls real tools on your filesystem and Python environment.

Built on LangChain v1 (`create_agent`) + LangGraph for persistent memory.

---

## Quota and model fallback

Gemini's free tier allows roughly **20 requests per day, tracked per model id**.
One agent turn costs 2-4 requests (a tool call is a separate request), so a
free key gives you around 5-10 turns before you hit the wall.

CRS works around this with a **fallback chain**. When the primary model
returns an error, the agent automatically switches to the next model in
`AGENT_MODEL_FALLBACKS` and keeps going. The banner shows the active chain.

| Order | Model | Notes |
| --- | --- | --- |
| primary | `gemini-3.5-flash-lite` | good balance of speed and tool-calling |
| fallback 1 | `gemini-3.1-flash-lite` | separate daily quota |
| fallback 2 | `gemini-flash-lite-latest` | separate daily quota |
| fallback 3 | `gemma-4-26b-a4b-it` | last resort, weakest tool-calling |

That is up to 4 independent daily quotas (~80 requests) on one key.

Change the chain in `.env`:

```ini
AGENT_MODEL="google_genai:gemini-3.5-flash-lite"
AGENT_MODEL_FALLBACKS="google_genai:gemini-3.1-flash-lite,google_genai:gemini-flash-lite-latest"
```

When every model is exhausted you get a precise report instead of a stack trace:

```
ERROR: Gemini quota exhausted (429).
  metric : GenerateRequestsPerDayPerProjectPerModel
  limit  : 20 requests per day
  resets : 9h38m3s
  fix    : use a paid key, switch model with --model, or wait for the reset.
```

Adding billing to the same key removes the cap entirely. That is the only
real fix for the ceiling; the fallback chain just buys you time.

---

## Quick start

```powershell
# 1. install dependencies (already done if you have the venv)
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. put your key in .env
#    GOOGLE_API_KEY="..."

# 3. launch it
.\crs.cmd
```

```
  ___    ___   ____        CRS  -  local AI agent
 / __|  / _ \ |  _ \       model:     google_genai:gemini-3.5-flash-lite
| (__  | |_) || |_) |      fallback:  gemini-3.1-flash-lite, gemini-flash-lite-latest, gemma-4-26b-a4b-it
 \___| |  _ < | ___) |      session:   default
      |_| \_\|____/        workspace: C:\django_project\LLM

workspace        C:\django_project\LLM
network          allowed
writes/deletes   allowed
shell tool       disabled
secret redaction on
approval gate    on (run_python, run_shell, delete_path)
audit log        C:\django_project\LLM\.agent\audit.log
```

Then type a task and press Enter.

```powershell
.\crs.cmd "Create a notes.md file, add a heading, and print it back"
```

### Run `crs` from anywhere

Put the project folder on your PATH once:

```powershell
$env:Path += ";C:\django_project\LLM"          # this session
setx PATH "%PATH%;C:\django_project\LLM"        # permanent, new terminals only
```

Then just type `crs`.

---

## Usage

| Command | What it does |
| --- | --- |
| `crs` / `.\crs.cmd` | Interactive chat with the agent |
| `crs "task"` | Run one task and exit |
| `python run.py --session work` | Use a named session (memory is per session) |
| `python run.py --model gemini-3.5-pro` | Override the model |
| `python run.py --workspace ./sandbox` | Restrict the agent to a different directory |
| `python run.py --shell` | Enable the `run_shell` tool |
| `python run.py --no-stream` | Disable token streaming |
| `python run.py --list-sessions` | List stored sessions |
| `python run.py --clear-session` | Wipe the current session's history |
| `python run.py --print-tools` | Print the tool schemas |
| `python run.py --selfcheck` | Offline test of every tool + the security rules |
| `python run.py --demo build-file` | Run a scripted demo with a mock model (no API key) |
| `python run.py --list-demos` | List the demos |

### Security flags

| Flag | Effect |
| --- | --- |
| `--read-only` | Block writes, edits and deletes |
| `--offline` | Block all outbound network access |
| `--no-approve` | Skip the approval prompt on risky tools |
| `--no-redact` | Stop masking secret-looking values |
| `--audit-log PATH` | Write the audit trail somewhere else |
| `--yes` | Auto-approve every gated tool call (scripting only) |

Inside the chat:

`/help` `/tools` `/policy` `/sessions` `/new` `/history` `/model` `/clear` `/exit`

---

## Tools

| Tool | Purpose |
| --- | --- |
| `read_file` | Read a text file (supports line ranges) |
| `write_file` | Create or overwrite a file |
| `edit_file` | Replace an exact substring (fails loudly on mismatch) |
| `list_directory` | List a directory |
| `make_directory` | Create a directory |
| `delete_path` | Delete a file or empty directory |
| `run_python` | Execute Python in a subprocess |
| `run_shell` | Execute a shell command (opt-in via `--shell`) |
| `calculate` | Evaluate math safely via AST (no `eval`) |
| `current_datetime` | Current time in any IANA timezone |
| `fetch_url` | Download a page as readable text |
| `web_search` | Web search (needs `TAVILY_API_KEY` or `SERPER_API_KEY`) |
| `write_todos` | Track multi-step plans (added by `TodoListMiddleware`) |

Filesystem tools are **sandboxed to the workspace**. Escapes are rejected unless
`AGENT_ALLOW_OUTSIDE_WORKSPACE=true`.

---

## Configuration (`.env`)

```ini
GOOGLE_API_KEY="your-gemini-key"

# optional - web search
TAVILY_API_KEY=""
SERPER_API_KEY=""

# optional - agent tuning
AGENT_NAME="CRS"
AGENT_MODEL="google_genai:gemini-3.5-flash"
AGENT_WORKSPACE="."
AGENT_MAX_STEPS=25
AGENT_TEMPERATURE=0.2

# security
AGENT_REQUIRE_APPROVAL=true
AGENT_ALLOW_NETWORK=true
AGENT_ALLOW_DESTRUCTIVE=true
AGENT_REDACT_SECRETS=true
AGENT_ALLOW_SHELL=false
AGENT_ALLOW_OUTSIDE_WORKSPACE=false
# AGENT_AUDIT_LOG=".agent/audit.log"
```

---

## Layout

```
run.py                  entry point
crs.cmd                 launcher (add the folder to PATH and type "crs")
agent/
  config.py             Settings dataclass, reads .env
  builder.py            model + system prompt + middleware -> compiled agent
  security.py           security policy: paths, redaction, SSRF, audit
  memory.py             SQLite-backed conversation memory (per session)
  cli.py                argument parsing, streaming, slash commands, approvals
  selfcheck.py          offline tool + security tests
  mock.py               scripted model for offline demos
  tools/
    __init__.py         tool registry
    filesystem.py       read/write/edit/list/delete
    execution.py        run_python, run_shell
    utility.py          calculate, current_datetime
    web.py              fetch_url, web_search
.agent/sessions.db      conversation history (created on first run)
.agent/audit.log        audit trail of every sensitive action
```

The agent graph is **model -> tools -> model -> ...** until it produces a final
answer, with these middleware layers:

- `TodoListMiddleware` - exposes `write_todos` for multi-step planning
- `SummarizationMiddleware` - compresses history past 70% of the context window
- `ModelRetryMiddleware` - exponential backoff on 429s and transient errors
- `ModelCallLimitMiddleware` - hard cap on model calls per turn
- `HumanInTheLoopMiddleware` - approval prompt before risky tools
- `wrap_tool_call` - redacts secrets from every tool result

---

## Security

A local agent with shell access is a loaded gun, so the defaults are defensive.

### What is protected

| Threat | Mitigation |
| --- | --- |
| Reading files outside the workspace | Paths are resolved and rejected if they escape the workspace |
| Stealing credentials | `.env`, ssh/aws dirs, `*.pem`, `id_rsa`, `credentials.json` etc. are blocked; secret-looking strings are masked before the model sees them |
| Leaking your API key | Subprocesses inherit a sanitized environment with all API keys removed |
| SSRF / cloud metadata theft | `fetch_url` refuses loopback, private ranges, `localhost` and `169.254.169.254`, and re-checks after redirects |
| Destructive commands | `rm -rf`, `mkfs`, `curl \| sh` and friends are rejected; `run_shell` is off by default |
| Unapproved damage | `run_python`, `run_shell` and `delete_path` pause for your approval |
| Prompt injection from files or web pages | Tool output is treated as data; the system prompt forbids following instructions found in content |
| Runaway loops | Step limit, recursion limit, per-call timeout, output size cap |

### Layers

1. **`agent/security.py`** - the policy: path guards, secret redaction, SSRF
   checks, shell deny-list, subprocess env sanitizing, audit log.
2. **Tool level** - every tool enforces the policy and returns a clear
   `ERROR: ...` instead of failing silently.
3. **Middleware level** - `HumanInTheLoopMiddleware` gates risky tools,
   a `wrap_tool_call` middleware redacts tool results as a second layer.
4. **Prompt level** - the system prompt marks file/web content as untrusted
   data and forbids secret handling and scope creep.
5. **Audit level** - every file write, delete, code execution, approval and
   block is recorded to `.agent/audit.log`.

### Deliberate limits

- The agent still runs as **you**, with your Windows account's permissions.
  The workspace sandbox stops *accidents*, not a determined attacker who has
  already broken out.
- `run_python` can still open a socket. Blocking all egress needs a container
  or a firewall rule, not Python.
- Prompt injection is mitigated, not solved. The approval gate is the real
  backstop: if the agent tries something destructive because a web page told
  it to, you get a prompt instead of a command.

---

## Extending it

**Add a tool** - drop a function in `agent/tools/`, register it in
`agent/tools/__init__.py`, done. Use the factory pattern so the tool can close
over `Settings`:

```python
from langchain_core.tools import tool

def my_tools(settings: Settings) -> list[BaseTool]:
    @tool
    def my_tool(arg: str) -> str:
        """One-line description the model sees."""
        return f"{arg} (workspace={settings.workspace})"
    return [my_tool]
```

**Change the personality** - edit `SYSTEM_PROMPT` in `agent/builder.py`.

**Add a different provider** - `AGENT_MODEL` accepts any LangChain model id
(`openai:gpt-4o`, `anthropic:claude-sonnet-4-5`, `ollama:llama3`), given the
matching env var is set.

---

## Notes and limits

- **The Gemini free tier is tiny.** For `gemini-2.5-flash` the free key allows
  roughly 5 requests/minute and **20 requests/day per project**. An agent turn
  burns 2-4 requests (one per tool round trip), so a free key dies after a
  handful of turns. Use a paid key for real work, or point `AGENT_MODEL` at a
  model with a bigger quota.
- The agent retries 429s with exponential backoff and reports the quota error
  clearly instead of failing silently. It also detects Gemini's "HTTP 200 with
  an empty body" quota response and retries it.
- `run_python` and `run_shell` execute on your machine with your permissions.
  Only enable `--shell` when you trust the request.
- History is stored unencrypted in `.agent/sessions.db`. It can contain
  whatever you (or the agent) wrote.
- `tzdata` is required for `current_datetime` on Windows, which has no system
  timezone database. It is in `requirements.txt`.
