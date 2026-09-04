# Qwen VLM integration

DVSA-API can consult an Azure-hosted **Qwen VLM** model as part of its
agentic-retrieval flow. Qwen is wired in as **just another function-tool**, a peer
of the AI-Search retrieval and the Perplexity fallback — the Foundry chat agent
invokes it the same way it invokes every other tool, and its answer is folded into
the single, consolidated reply returned to the user.

No Django models, ingestion/preprocessing, or existing API semantics change. There
is **no new endpoint**: Qwen reaches the user through the existing chat path
(`ChatAPIView` → `SessionAzureEnvironment.ask` → `FoundryAgents.synthesize_from_chat_agent`).

## How it fits in

`core/azure/analyzer.py` exposes `ask_qwen_vlm(query_text, account_id, video_id)`,
modelled on the existing `ask_perplexity` tool. It is registered into the Foundry
`FunctionTool` sets (`analyzer_functions()` / `image_user_functions()`), so the
consolidating chat agent can select it, pass it the user's question, and include
its response in the final narrative alongside the other tools' output.

```
ChatAPIView → env.ask() → FoundryAgents.synthesize_from_chat_agent()
                                    │
                                    ├─ AI-Search knowledge agent   (RAG retrieval)
                                    ├─ ask_perplexity              (multimodal fallback)
                                    └─ ask_qwen_vlm                (Qwen VLM)   ← added
```

## Global on/off (default: on)

Qwen participates in user queries **by default**. A single global setting turns it
off and restores the exact pre-Qwen behaviour (the tool sets become byte-for-byte
what they were before), so the integration is fully backward compatible.

| Setting / env var    | Default                                                                 | Meaning                                             |
| -------------------- | ----------------------------------------------------------------------- | --------------------------------------------------- |
| `DVSA_QWEN_ENABLED`  | `true`                                                                   | Global on/off. Set falsy (`0`/`false`/`no`/`off`) to disable Qwen entirely. |
| `DVSA_QWEN_API_KEY`  | _(unset)_                                                               | API key for the Azure Foundry endpoint.             |
| `DVSA_QWEN_ENDPOINT` | `https://found-vision-1.services.ai.azure.com/openai/v1/chat/completions` | OpenAI-compatible chat-completions endpoint.        |
| `DVSA_QWEN_MODEL`    | `qwen--qwen3.5-0.8b`                                                     | Model id passed in the request body.                |

When `DVSA_QWEN_ENABLED` is off, or when `DVSA_QWEN_API_KEY` is unset, `ask_qwen_vlm`
returns a benign `"No comment."` and makes **no network call** — matching how the
Perplexity tool degrades without credentials.

### Setting the key

```bash
export DVSA_QWEN_API_KEY="<your-azure-foundry-key>"
# optional overrides
export DVSA_QWEN_ENABLED=true
export DVSA_QWEN_ENDPOINT="https://found-vision-1.services.ai.azure.com/openai/v1/chat/completions"
export DVSA_QWEN_MODEL="qwen--qwen3.5-0.8b"
```

## Request / response

The tool sends an OpenAI-style chat request with the recommended generation
settings (`temperature=0.6`, `top_p=0.95`):

```http
POST /openai/v1/chat/completions
Authorization: Bearer $DVSA_QWEN_API_KEY
Content-Type: application/json

{
  "model": "qwen--qwen3.5-0.8b",
  "messages": [
    {"role": "system", "content": "You are an aerial drone image and vision analyst."},
    {"role": "user", "content": "count the vehicles at the interchange"}
  ],
  "temperature": 0.6,
  "top_p": 0.95,
  "stream": false
}
```

The response's `choices[0].message.content` becomes the tool's answer:

```json
{ "choices": [ { "message": { "role": "assistant", "content": "two cars and a truck" } } ] }
```

which the chat agent merges into the consolidated reply delivered to the user.

## Testing

`tests/test_qwen_tool.py` runs fully offline. It mocks the endpoint with
`requests_mock` to assert the request contract (endpoint, bearer auth, message
shape, generation params) and response parsing, verifies graceful fallback on HTTP
errors and timeouts, and asserts the global flag toggles tool registration —
including that the tool sets are unchanged when Qwen is disabled.

CI (`.github/workflows/ci.yml`) sets a mock-only `DVSA_QWEN_API_KEY` so the
configured path is exercised without any live call.
