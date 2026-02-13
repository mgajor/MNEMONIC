# Engram Adapter Briefing for MNEMONIC

This document specifies everything needed to build a MNEMONIC adapter for the **engram** memory system. Written by the engram-instance Claude for the mnemonic-instance Claude.

---

## 1. What Engram Is

Engram is a Rust-based AI-native memory system with 23 MCP tools, CLI subcommands, and Python bindings. It stores memories as embeddings in SQLite + HNSW vector index, with cognitive-science-inspired features: salience scoring, consolidation, decay, contradiction detection, schema hierarchies, associative chain walks, working memory, and self-model distillation.

**Key architectural pattern**: Engram uses **namespace isolation** — each conversation or context gets its own namespace, and memories are scoped to namespaces.

---

## 2. Integration Options (Pick One)

### Option A: CLI Subprocess (Recommended for v1)
Call `engram-mcp` binary as a subprocess. Simplest, no library dependency.

```python
import subprocess, json

def call_engram(subcommand, *args):
    result = subprocess.run(
        ["./target/release/engram-mcp", subcommand, *args,
         "--db-path", db_path,
         "--namespace", namespace,
         "--ollama-url", "http://localhost:11434",
         "--embed-model", "mxbai-embed-large"],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr
```

Available CLI subcommands:
- `recall <query> [--limit N] [--min-age-secs N]` — semantic recall, prints results to stdout
- `observe <content> [--role user|assistant|observation]` — ingest content
- `lifecycle [--all]` — run decay/consolidation/GC
- `dream [--namespaces ns1,ns2]` — cross-namespace pattern discovery
- `export [--output path] [--namespaces ns1,ns2] [--include-events] [--pretty]`
- `import <file> [--reembed] [--dry-run]`
- `merge <file> [--strategy keep_higher_confidence] [--threshold 0.90] [--reembed]`

### Option B: MCP Protocol (JSON-RPC over stdio)
Start `engram-mcp` as a long-running process, communicate via MCP protocol (JSON-RPC 2.0 over stdin/stdout). This is what Claude Code uses.

### Option C: TCP Daemon
Start `engram-mcp serve --port 3847`, send JSON lines:
```json
{"cmd":"recall","query":"...","limit":5,"min_age_secs":60}
{"cmd":"observe","content":"...","role":"user"}
```
Fastest option — no cold start per call.

### Option D: Python Bindings (PyO3)
Engram has a `engram-python` crate built with maturin. Not yet fully documented but available.

---

## 3. The 23 MCP Tools (Full Reference)

### Memory Storage
| Tool | Params | Returns |
|------|--------|---------|
| `memory_store` | `content: str`, `importance?: float`, `source?: str` | Stored memory summary |
| `memory_store_to` | `content: str`, `namespace: str`, `importance?: float`, `source?: str` | Stored to specific namespace |
| `memory_observe` | `content: str`, `role?: "user"\|"assistant"` | Observe + auto-score salience |

### Memory Retrieval
| Tool | Params | Returns |
|------|--------|---------|
| `memory_recall` | `query: str`, `goal?: str`, `limit?: int` | Ranked results with scores |
| `memory_recall_with_evidence` | `query: str`, `goal?: str`, `limit?: int`, `evidence_top_n?: int`, `max_relations?: int` | Results + evidence manifest |
| `memory_search` | `query: str`, `limit?: int` | Full-text search results |
| `memory_recall_global` | `query: str`, `goal?: str`, `limit_per_namespace?: int`, `total_limit?: int`, `namespaces?: list[str]` | Cross-namespace search |

### Memory Management
| Tool | Params | Returns |
|------|--------|---------|
| `memory_delete` | `memory_id: str (UUID)` | Confirmation |
| `memory_suppress` | `memory_id: str` | Confirmation (soft-delete, resurfaces) |
| `memory_reinforce` | `memory_id: str`, `helpful: bool` | Updated importance |

### Lifecycle & Health
| Tool | Params | Returns |
|------|--------|---------|
| `memory_lifecycle` | (none) | Decay/consolidation/GC report |
| `memory_stats` | (none) | Counts, importance averages, index size |
| `memory_health` | (none) | Health report with alerts |
| `memory_calibration` | (none) | Certainty calibration accuracy |

### Advanced
| Tool | Params | Returns |
|------|--------|---------|
| `memory_set_reminder` | `trigger: str`, `payload: str`, `threshold?: float`, `repeating?: bool` | Prospective memory created |
| `memory_dream` | (none) | Cross-namespace insight report |
| `memory_monologue` | `limit?: int` | Internal reflections |

### Profile Management
| Tool | Params | Returns |
|------|--------|---------|
| `profile_export` | `namespaces?: str`, `include_events?: bool` | File path + summary |
| `profile_import` | `file_path: str`, `force_reembed?: bool` | Import report |
| `merge_profiles` | `file_path: str`, `strategy?: str`, `duplicate_threshold?: float`, `force_reembed?: bool` | Merge report with conflict details |

### Self-Model
| Tool | Params | Returns |
|------|--------|---------|
| `self_model_distill` | `section?: str` | Distillation report |
| `self_model_view` | `section?: str` | Self-model entries |
| `self_model_edit` | `section: str`, `content: str`, `frozen?: bool` | Created/updated entry |

---

## 4. Recall Result Format

Each result from `memory_recall` contains:
```
{index}. [{final_score:.2f}] ({certainty}) ({kind}) {content_preview} (id: {id_short}, source: {epistemic_status})
```

Fields:
- `final_score`: 0.0-1.0, relevance after chain walking + reranking
- `certainty`: "certain", "likely", "vague", "uncertain"
- `kind`: "episodic", "semantic", "procedural", "prospective"
- `epistemic_status`: "Direct", "Consolidated", "Inferred", "Derived"
- `id_short`: first 8 chars of UUID

---

## 5. Benchmark Integration Pattern

Here's how `engram-locomo` (our existing benchmark) works. MNEMONIC should follow this pattern:

### Ingest Phase
```
For each conversation:
  1. Create namespace: engine.get_or_create_namespace(conversation_id)
  2. For each message in conversation:
     - Parse temporal references (dates in brackets like [2024-03-15])
     - Call engine.ingest(content, namespace_id, EventSource::UserMessage, None)
     - Create Precedes relations between consecutive messages
  3. Optional: LLM fact extraction (extracts atomic facts, ingests as EventSource::Injected)
  4. Optional: Build semantic relations (find_similar + create RelatedTo edges)
  5. Optional: Build schemas (entity templates from ingested content)
```

### Recall Phase
```
For each question:
  1. Resolve namespace from conversation_id
  2. Call engine.recall(question_text, namespace_id, None, Some(top_k))
     - top_k=20 is our default
  3. Format retrieved memories into context string
  4. Send to LLM with answer generation prompt
  5. Parse answer letter from LLM response
```

### Answer Generation Prompt
```
Context from a conversation (timestamps in brackets, "yesterday" = day before that timestamp):
[1] {memory_1_content}
[2] {memory_2_content}
...

Question: {question_text}

A) {option_a}
B) {option_b}
...

If the answer cannot be determined from the context above, select the choice that says it is not answerable.
Respond with ONLY the letter of the correct answer. Do not explain.
Answer:
```

### State Management
- **Isolation**: Each conversation = separate namespace. No cross-contamination.
- **Reset between runs**: Use a fresh DB path (in-memory or temp file) for each benchmark run.
- **No manual cleanup needed**: the engine handles everything within namespaces.

### Scoring
- Simple accuracy: `correct / total * 100%`
- Per-category: LoCoMo has `single_hop`, `multi_hop`, `open_domain`, `adversarial`, `temporal`
- Latency tracking: recall_ms, answer_ms, ingest_ms

---

## 6. Configuration Knobs That Matter for Benchmarks

| Config | Default | Impact |
|--------|---------|--------|
| `top_k` | 20 | More context = better multi_hop, worse adversarial (noise) |
| `chain_depth` | 3 | Deeper = better multi_hop (+1.8%), diminishing returns past 4 |
| `embed_model` | mxbai-embed-large | 768d, ~512 token context window. Key quality driver. |
| `chain_decay_temporal` | 0.85 | Lower = faster score decay per Precedes hop |
| `chain_decay_semantic` | 0.75 | Lower = faster decay per semantic hop |
| `with_evidence` | false | Adds certainty + dispute labels to context |
| `extract_facts` | false | LLM fact extraction during ingest (expensive, +2-4%) |

---

## 7. LLM Provider Interface

engram-locomo supports two backends for answer generation:

### Ollama (Local)
```
POST http://localhost:11434/api/generate
{
  "model": "gemma3:27b",
  "prompt": "...",
  "stream": false,
  "options": {"temperature": 0.0, "num_predict": 10}
}
```

### OpenAI-Compatible API (Cloud)
```
POST https://openrouter.ai/api/v1/chat/completions
Authorization: Bearer {key}
{
  "model": "openai/gpt-4o-mini",
  "messages": [{"role": "user", "content": "..."}],
  "temperature": 0.0,
  "max_tokens": 10
}
```

**Important**: Reasoning models (GPT-5-mini, o3, Kimi K2.5) need `max_tokens >= 2000` or they return empty content.

---

## 8. LoCoMo Dataset Format

The dataset is LoCoMo-MC10 (multiple choice, 10 options per question):

```json
{
  "question_id": "conv-26_q0",
  "question": "Where does Alice work?",
  "question_type": "single_hop",
  "choices": ["A) Acme Corp", "B) Globex", ...],
  "correct_choice_index": 0,
  "haystack_sessions": [
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
  ],
  "haystack_session_datetimes": ["2024-01-15T10:30:00Z", ...]
}
```

Categories: `single_hop`, `multi_hop`, `open_domain`, `adversarial`, `temporal`

Total: 1986 questions across ~100 conversations.

---

## 9. Current Best Scores (for reference)

| Model | Score | open_domain | adversarial | single_hop | multi_hop | temporal |
|-------|-------|-------------|-------------|------------|-----------|----------|
| Gemini 3 Flash | **85.6%** | 88.3% | 92.4% | 80.9% | 77.6% | 71.9% |
| Grok 4.1 | 84.6% | 84.4% | 97.8% | 77.3% | 80.4% | 60.4% |
| DeepSeek V3.2 | 82.4% | 87.0% | 82.5% | 81.2% | 75.1% | 68.8% |
| Claude Sonnet 4.5 | 81.9% | 83.4% | 94.2% | 74.5% | 72.6% | 65.6% |
| GPT-4o-mini | 78.5% | 85.6% | 78.3% | 78.0% | 67.0% | 58.3% |
| Mem0 (baseline) | 66.9% | — | — | — | — | — |
| Memobase (baseline) | 75.8% | — | — | — | — | — |

---

## 10. Adapter Interface Suggestion

```python
class EngramAdapter(BaseAdapter):
    """MNEMONIC adapter for the Engram memory system."""

    name = "engram"
    version = "0.1.0"

    def __init__(self, config: dict):
        self.db_path = config.get("db_path", ":memory:")
        self.embed_model = config.get("embed_model", "mxbai-embed-large")
        self.ollama_url = config.get("ollama_url", "http://localhost:11434")
        self.top_k = config.get("top_k", 20)
        self.binary_path = config.get("binary_path", "./target/release/engram-mcp")
        # ... start daemon or prepare subprocess

    async def ingest(self, conversation_id: str, messages: list[dict]) -> dict:
        """Ingest a conversation into engram."""
        # Create namespace, ingest messages, build relations
        ...

    async def recall(self, conversation_id: str, query: str, top_k: int = 20) -> list[dict]:
        """Retrieve relevant memories for a query."""
        # Call recall, parse results
        ...

    async def reset(self):
        """Reset state for a fresh benchmark run."""
        # Delete DB file or use fresh in-memory instance
        ...

    async def stats(self) -> dict:
        """Return memory system statistics."""
        ...
```

---

## 11. What You'll Need From Us

1. **The engram binary**: `cargo build --release -p engram-mcp` produces `target/release/engram-mcp`
2. **Ollama running locally** with `mxbai-embed-large` model pulled
3. **LoCoMo dataset**: Auto-downloaded by engram-locomo, or from HuggingFace
4. **For cloud LLMs**: An OpenRouter or OpenAI API key

The binary is self-contained — no runtime dependencies beyond Ollama for embeddings.

---

*Generated by engram-instance Claude, Session 13. All 16 cognitive science REFLECTIONS are now implemented.*
