# Connecting Claude Code to vLLM + llm-d: Step-by-Step Guide

## How It Works

```
┌────────────┐    /v1/messages     ┌──────────┐    internal     ┌───────────┐
│ Claude Code │ ──────────────────→ │   vLLM   │ ─────────────→ │ GLM 5.2   │
│  (client)   │  Anthropic Messages │ (server)  │  glm47/glm45  │  (model)  │
└────────────┘    API format       └──────────┘    parsers      └───────────┘
                                        │
                                        │  also serves
                                        ↓
                               /v1/chat/completions
                                 (OpenAI format)
                                        │
                               ┌────────────────┐
                               │  inference-perf │
                               │  (load gen)     │
                               └────────────────┘
```

**Claude Code only speaks the Anthropic Messages API** (`/v1/messages`). It does NOT use `/v1/chat/completions`. Setting `OPENAI_BASE_URL` has no effect on Claude Code.

**vLLM serves both API formats** from the same process. GLM-5.2 requires vLLM v0.23.0+, which includes the `/v1/messages` endpoint and the unified `Parser.parse()` interface sharing the `glm47` tool-call parser and `glm45` reasoning parser.

**inference-perf uses the OpenAI API** (`/v1/chat/completions`) for load generation. This is a different client hitting the same vLLM server.

---

## Prerequisites

- vLLM v0.23.0+ (minimum for GLM-5.2 model support)
- GLM 5.2 deployed via vLLM with tool-call parsers enabled
- `kubectl` access to the cluster (or direct network access to vLLM)
- Claude Code CLI installed (`npm install -g @anthropic-ai/claude-code` or via [claude.com/code](https://claude.com/code))

## Step 1: Verify vLLM is serving both APIs

After deploying vLLM with GLM 5.2 (see [single-node practical guide](glm-5.2-single-node-practical-guide.md)), verify both endpoints work.

**Port-forward to your local machine:**

```bash
# If using llm-d router:
kubectl port-forward -n $NAMESPACE service/${GUIDE_NAME}-epp 8000:80

# If hitting vLLM directly (no router):
kubectl port-forward -n $NAMESPACE pod/vllm-0 8000:8000
```

**Test the OpenAI API** (`/v1/chat/completions`):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2-fp8",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }' | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'][:100])"
```

**Test the Anthropic Messages API** (`/v1/messages`):

```bash
curl -s http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: dummy" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "glm-5.2-fp8",
    "max_tokens": 50,
    "messages": [{"role": "user", "content": "Hello"}]
  }' | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['content'][0]['text'][:100])"
```

Both should return model output. If `/v1/messages` fails with 404, your vLLM version may be too old — GLM-5.2 requires v0.23.0+.

## Step 2: Create the Claude Code environment file

Create a `claude.env` file with these variables:

```bash
# claude.env — Claude Code → vLLM connection via Anthropic Messages API
#
# Claude Code ONLY speaks /v1/messages (Anthropic format).
# It ignores OPENAI_BASE_URL entirely.
# vLLM translates Messages API requests internally to drive the model.

ANTHROPIC_BASE_URL=http://localhost:8000
ANTHROPIC_API_KEY=dummy
ANTHROPIC_MODEL=glm-5.2-fp8
ANTHROPIC_SMALL_FAST_MODEL=glm-5.2-fp8
CLAUDE_CODE_USE_VERTEX=0
```

**Variable reference:**

| Variable | Required | What it does |
|----------|----------|-------------|
| `ANTHROPIC_BASE_URL` | Yes | Points Claude Code at vLLM's `/v1/messages` endpoint |
| `ANTHROPIC_API_KEY` | Yes | Any non-empty string (vLLM doesn't validate by default) |
| `ANTHROPIC_MODEL` | Yes | Must match vLLM's `--served-model-name` |
| `ANTHROPIC_SMALL_FAST_MODEL` | Yes | Used for background tasks (sub-agents, haiku-tier work). Set to same model if you only have one |
| `CLAUDE_CODE_USE_VERTEX` | No | Set to `0` to disable Vertex AI auth (avoids auth errors) |

**What does NOT work:**

| Variable | Why it fails |
|----------|-------------|
| `OPENAI_BASE_URL` | Claude Code ignores this entirely |
| `OPENAI_API_KEY` | Claude Code ignores this entirely |

## Step 3: Launch Claude Code

```bash
# Source the env file and launch
source claude.env && claude
```

Or equivalently:

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_MODEL=glm-5.2-fp8 \
ANTHROPIC_SMALL_FAST_MODEL=glm-5.2-fp8 \
CLAUDE_CODE_USE_VERTEX=0 \
claude
```

## Step 4: Verify tool calling works

Once Claude Code starts, test these prompts:

| Test | Prompt | What to look for |
|------|--------|-----------------|
| File read | `What's in this directory?` | Claude Code calls its `Bash` tool to run `ls`, model returns tool call via `/v1/messages` |
| File write | `Create a file called hello.py that prints hello world` | Model generates a `Write` tool call with file content |
| Multi-turn | `Now add a main guard to hello.py` | Model reads the file it just created, modifies it |
| Reasoning | `Explain how the code works, think step by step` | Model returns reasoning content (extracted by `glm45` parser) alongside the answer |

If tool calls fail silently (model responds with text instead of tool calls), check:
- vLLM has `--enable-auto-tool-choice` and `--tool-call-parser glm47`
- The model name in `ANTHROPIC_MODEL` exactly matches `--served-model-name`
- vLLM logs show requests arriving at `/v1/messages` (not 404s)

## Step 5: Persistent configuration (optional)

Instead of sourcing `claude.env` every time, add to `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8000",
    "ANTHROPIC_API_KEY": "dummy",
    "ANTHROPIC_MODEL": "glm-5.2-fp8",
    "ANTHROPIC_SMALL_FAST_MODEL": "glm-5.2-fp8",
    "CLAUDE_CODE_USE_VERTEX": "0"
  }
}
```

**Warning:** This overrides your normal Anthropic API connection. Remove these settings when you want to use Claude Code with Anthropic's API again.

## Step 6: Enable prefix caching (performance)

Claude Code sends the same system prompt on every turn. With prefix caching enabled on vLLM, repeated prefixes are served from KV cache instead of recomputed.

Ensure vLLM was started with:
```bash
--enable-prefix-caching
```

From Claude Code v2.1.181+, the system prompt attribution block is stable across requests, so prefix caching works correctly. On older versions, disable attribution to prevent cache-busting:

```bash
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
```

## Troubleshooting

### "Model not found" error
The model name in `ANTHROPIC_MODEL` must exactly match vLLM's `--served-model-name`. Check with:
```bash
curl -s http://localhost:8000/v1/models | python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data']]"
```

### Claude Code hangs or times out
vLLM may still be loading the model. Check readiness:
```bash
curl -s http://localhost:8000/health
```

### Tool calls return as plain text
The model is generating tool-call-like text but vLLM isn't parsing it. Verify:
- `--tool-call-parser glm47` is set (for GLM 5.2)
- `--enable-auto-tool-choice` is set
- Check vLLM logs for parser errors

### "API key invalid" error
Set `ANTHROPIC_API_KEY` to any non-empty string. vLLM's Anthropic endpoint checks for the header's presence but doesn't validate the value by default.

### MTP breaks tool calling
Known issue ([vLLM #41967](https://github.com/vllm-project/vllm/issues/41967)). If tool-call arguments are truncated with MTP enabled, restart vLLM without `--speculative-config` flags and test again. Report the issue upstream if it reproduces.

---

## Architecture Summary

```
                        vLLM Process (single server)
                       ┌─────────────────────────────┐
                       │                             │
  Claude Code          │  /v1/messages               │
  (ANTHROPIC_BASE_URL) │  ← Anthropic Messages API   │
  ────────────────────→│  → AnthropicServingMessages  │
                       │  → Internal translation      │──→ GLM 5.2
  inference-perf       │  → glm47 tool parser         │    (model)
  (server.base_url)    │  → glm45 reasoning parser    │
  ────────────────────→│                             │
                       │  /v1/chat/completions        │
                       │  ← OpenAI Chat API           │
                       │  → OpenAIServingChat         │
                       │  → Same parser path          │
                       └─────────────────────────────┘
```

GLM-5.2 requires vLLM v0.23.0+. Both `/v1/messages` and `/v1/chat/completions` go through the same unified `Parser.parse()` interface. The Anthropic Messages API handler translates the request format, but the model execution, tool-call parsing, and reasoning extraction are identical.
