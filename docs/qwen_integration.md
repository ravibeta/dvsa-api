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
| `DVSA_QWEN_BACKEND`  | `azure`                                                                 | `azure` calls the Foundry endpoint; `onnx` runs Qwen3.5-0.8B locally (see below). |
| `DVSA_QWEN_API_KEY`  | _(unset)_                                                               | API key for the Azure Foundry endpoint (`azure` backend). |
| `DVSA_QWEN_ENDPOINT` | `https://found-vision-1.services.ai.azure.com/openai/v1/chat/completions` | OpenAI-compatible chat-completions endpoint (`azure` backend). |
| `DVSA_QWEN_MODEL`    | `qwen--qwen3.5-0.8b`                                                     | Model id passed in the request body (`azure` backend). |
| `DVSA_QWEN_ONNX_MODEL_PATH` | _(unset)_                                                        | Local Qwen3.5-0.8B ONNX model directory (`onnx` backend). |

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

## Standalone / local mode (ONNX, no Azure)

For local deployments that must run **without any Azure dependency**, set the
backend to `onnx` and point at a `Qwen3.5-0.8B` model exported to ONNX. The tool
then runs inference on the box via
[`onnxruntime-genai`](https://github.com/microsoft/onnxruntime-genai) instead of
calling the Foundry endpoint — no key, no network:

```bash
export DVSA_QWEN_BACKEND=onnx
export DVSA_QWEN_ONNX_MODEL_PATH="/models/qwen3.5-0.8b-onnx"
pip install onnxruntime-genai   # opt-in; only needed for real local inference
```

The local runner lives in `core/azure/qwen_onnx.py`. It uses the Qwen ChatML
template with the recommended generation settings (`temperature=0.6`,
`top_p=0.95`) and returns the assistant reply, which the chat agent folds into the
consolidated answer exactly as with the Azure backend — the routing is invisible
to the rest of the pipeline. If the model path is unset, `onnxruntime-genai` is not
installed, or generation fails, the tool degrades to `"No comment."` just like the
Azure path, so the pipeline stays runnable while a box is still being provisioned.

The `onnxruntime-genai` interaction is encapsulated behind an injectable
`generator_factory`, so the routing and fallback logic are covered by fully offline
tests — no ONNX weights or runtime are required in CI.

## Baseline comparison endpoint (`/baseline-test`, Ollama)

To compare the agentic chat answer against a **raw** VLM answer, there is a
peer of the chat endpoint that bypasses all DVSA agentic/RAG synthesis and
returns only what a locally hosted `qwen2.5vl:7b` says via
[Ollama](https://ollama.com):

```
PUT /api/v1/videos/baseline-test/     (IsAuthenticated, same as chat/)
form-data: account_id, query, image (optional)
→ {"text": "<qwen2.5vl:7b answer>", "imageUrl": null, "downloadUrl": null}
```

It takes the **same request and returns the same response shape** as
`PUT /api/v1/videos/videos/chat/`, so the two answers drop straight into one
comparison harness — the only difference is that `baseline-test` sends the
question (and optional image) directly to the model instead of running the
agent. The call reproduces `local-serve-and-query-qwen.py`
(`ollama.Client(host).chat(model="qwen2.5vl:7b", ...)`, deterministic
`temperature=0.0`, `num_ctx=8192`) in `core/azure/qwen_ollama.py`.

Provision the local model, then point the endpoint at it:

```bash
ollama pull qwen2.5vl:7b
export OLLAMA_HOST="127.0.0.1:8848" && ollama serve   # in its own shell
export DVSA_OLLAMA_HOST="http://localhost:8848"       # default
export DVSA_OLLAMA_QWEN_MODEL="qwen2.5vl:7b"          # default
pip install ollama                                    # opt-in; only for real inference
```

| Setting / env var         | Default                  | Meaning                                             |
| ------------------------- | ------------------------ | --------------------------------------------------- |
| `DVSA_OLLAMA_HOST`        | `http://localhost:8848`  | Base URL of the local Ollama server.                |
| `DVSA_OLLAMA_QWEN_MODEL`  | `qwen2.5vl:7b`           | Ollama model tag served at `/baseline-test`.        |

If Ollama is unreachable, the `ollama` package is missing, or generation fails,
the endpoint degrades to `"No comment."` — like the Azure and ONNX paths — so it
stays callable while a box is still being provisioned. The Ollama interaction is
encapsulated behind an injectable `client_factory`, so the routing and fallback
logic are covered by fully offline tests (`tests/test_baseline_test.py`), with no
Ollama server or package required in CI.

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

`tests/test_qwen_tool.py` runs fully offline. It mocks the network with
`unittest.mock` (no extra test dependency) to assert the Azure request contract
(endpoint, bearer auth, message shape, generation params) and response parsing,
verifies graceful fallback on HTTP errors and timeouts, checks that the `onnx`
backend routes to the local runner (with no HTTP call) and degrades to
`"No comment."` when unconfigured, and asserts the global flag toggles tool
registration — including that the tool sets are unchanged when Qwen is disabled.

CI (`.github/workflows/ci.yml`) sets a mock-only `DVSA_QWEN_API_KEY` so the
configured path is exercised without any live call.
