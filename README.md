# theword

**Local, privacy-first voice dictation with persistent graph memory.**

theword captures audio from your microphone, transcribes it with a local Whisper model, optionally rewrites it with a small LLM, and types the result wherever your cursor is. Everything it learns about your vocabulary and writing style is stored in a SQLite-backed memory graph that grows smarter over time.

> **Built on [cede](https://github.com/MikeSquared-Agency/cede)**, which itself is built on [cortex-embedded](https://github.com/MikeSquared-Agency/cortex-embedded).

## Ecosystem

```
cortex-embedded          <-- embedded memory graph engine (upstream)
  |-- cede               <-- forkable agent starter kit
       |-- theword        <-- you are here (dictation + graph memory)
```

## What You Get

- **Voice dictation** — hold a hotkey (default: AltGr), speak, release; text is typed at your cursor
- **Local Whisper STT** — whisper.cpp via whisper-rs; models from `ggml-tiny.en` to `ggml-medium`
- **WebRTC VAD** — speech boundary detection; no fixed recording length
- **LLM transcript rewrite** — remove filler words, fix punctuation, detect commands (optional; works with Ollama or Anthropic)
- **Vocab learning** — corrections are persisted as zero-decay `VocabEntry` nodes; the rewrite LLM is briefed with them
- **Embedded memory graph** — 20 node kinds, 6 edge kinds, full provenance tracking
- **Graph-native memory** — fresh HNSW-based briefing per turn, no growing message history
- **Hybrid recall** — semantic search + graph traversal + recency window (last 7 messages always included)
- **Local embeddings** — BAAI/bge-small-en-v1.5 via fastembed (384-dim, no API calls)
- **Auto-linking + contradiction detection** — three-tier: cosine → negation keywords → LLM adjudication
- **Importance decay + trust propagation** — beliefs strengthen or weaken over time
- **LLM backends** — Anthropic Claude or Ollama (local)
- **Tool registry** — custom tools that write results into the graph with provenance
- **Sub-agents** — delegate tasks to scoped sub-agents in the shared graph
- **TUI graph explorer** — interactive terminal UI with chat, node inspection, visualization
- **Floating GUI overlay** — always-on-top microphone button; click to record, VAD ends the utterance, text is typed automatically
- **CLI** — dictate, chat, ask, memory search, vocab management, soul/identity, graph visualization, diagnostics

## Quick Start

```bash
git clone https://github.com/MikeSquared-Agency/theword.git
cd theword

# Build (requires C++ toolchain for whisper.cpp)
cargo build --release

# Initialize database and download Whisper model (~150 MB for ggml-base.en)
theword init

# Floating GUI overlay (click mic to record, always on top)
theword gui
ANTHROPIC_API_KEY=sk-ant-... theword gui

# Start the hotkey dictation daemon (main mode)
# Hold AltGr to record, release to transcribe and type
ANTHROPIC_API_KEY=sk-ant-... theword listen
# or with a local LLM for the rewrite pass:
theword --ollama qwen2.5:1.5b listen

# Record a single utterance (no hotkey, for testing)
theword dictate

# Interactive chat session (uses graph memory)
ANTHROPIC_API_KEY=sk-ant-... theword chat
theword --ollama qwen2.5:1.5b chat

# Single query
theword ask "What do you know?"

# Interactive graph explorer
ANTHROPIC_API_KEY=sk-ant-... theword graph explore

# Graph overview
theword graph overview

# Memory operations
theword memory stats
theword memory search "topic"
theword memory show <node_id>

# Vocabulary corrections
theword vocab list
theword vocab add "sea dee" "cede"
theword vocab remove "sea dee"

# Identity
theword soul show
theword soul edit

# Diagnostics
theword doctor
theword consolidate
```

## How It Works

### Dictation Pipeline

```
microphone
  → cpal (16 kHz mono PCM)
  → WebRTC VAD (30 ms frames, silence detection)
  → Whisper STT (local ggml model via whisper-rs)
  → LLM rewrite (filler removal, punctuation, command detection)
  → enigo (keyboard simulation) / arboard (clipboard)
```

The rewrite LLM is given a context briefing from the graph — your soul nodes, recent vocab corrections, and semantically similar past turns — so it learns your writing style and domain vocabulary over time.

### Memory Graph

Every dictation turn, vocabulary correction, chat session, and tool call is a node in a single SQLite-backed graph. Nodes are connected by typed edges (RelatesTo, Contradicts, Supports, DerivesFrom, PartOf, Supersedes). Semantic search uses a 2-tier HNSW index backed by 384-dim local embeddings.

## Tuning

### Dictation Config

Edit `DictationConfig::default()` in `src/config.rs`:

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `whisper_model` | `BaseEn` | Whisper model size (Tiny/Base/Small/Medium) |
| `language` | `"en"` | BCP-47 hint for Whisper; `None` = auto-detect |
| `vad_mode` | `Aggressive` | WebRTC VAD aggressiveness |
| `silence_threshold_ms` | 700 | Milliseconds of silence that end an utterance |
| `min_speech_ms` | 300 | Discard clips shorter than this |
| `hotkey.key` | `"AltGr"` | Trigger key (`AltGr`, `Alt`, `F9`–`F12`) |
| `hotkey.hold_to_talk` | `true` | Hold = record; `false` = toggle |
| `rewrite_enabled` | `true` | Run transcript through LLM for cleanup |
| `rewrite_model` | `"qwen2.5:1.5b"` | Ollama model for the rewrite pass |
| `output_method` | `Type` | `Type` / `Clipboard` / `Both` |
| `learn_corrections` | `true` | Persist vocab corrections to graph |

### Core Agent Config

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `session_recency_window` | 7 | Recent messages always included in briefing |
| `auto_link_cosine_threshold` | 0.75 | Minimum similarity for auto-linking |
| `contradiction_cosine_threshold` | 0.85 | Trigger contradiction detection |
| `max_iterations` | 10 | Agent loop iteration limit |
| `decay_interval_secs` | 60 | Background decay sweep frequency |

## Architecture

```
+---------------------------------------------------+
|                    theword                         |
+------------+----------+-----------+---------------+
|  dictation | recall   | briefing  |  agent        |
| (audio →   | (HNSW +  | (scored   | (loop +       |
|  VAD →     |  graph)  |  context) |  subagent)    |
|  stt →     |          |           |               |
|  rewrite → |          |           |               |
|  output)   |          |           |               |
+------------+----------+-----------+---------------+
|                  graph + memory                    |
|           (BFS, scoring, decay)                    |
+---------+------------------------------------------+
|  HNSW   |              SQLite                      |
| (2-tier)|       (WAL, bundled rusqlite)            |
+---------+------------------------------------------+
|            fastembed                               |
|      (BAAI/bge-small-en-v1.5)                      |
+----------------------------------------------------+
```

## Tests

```bash
# Run all tests
cargo test -- --test-threads=1
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| `rusqlite` (bundled) | SQLite with WAL mode |
| `instant-distance` | HNSW approximate nearest neighbor |
| `fastembed` | Local text embeddings (ONNX) |
| `tokio` | Async runtime |
| `reqwest` | HTTP client for Anthropic |
| `clap` | CLI argument parsing |
| `ratatui` + `crossterm` | TUI graph explorer |
| `async-channel` | Background task channels |
| `cpal` | Cross-platform audio capture |
| `ringbuf` | Lock-free ring buffer for audio stream |
| `webrtc-vad` | Voice activity detection |
| `whisper-rs` | Whisper.cpp bindings for local STT |
| `enigo` | Keyboard simulation (typing output) |
| `rdev` | Global hotkey listener |
| `arboard` | System clipboard access |
| `eframe` + `egui` | Floating GUI overlay |
| `ureq` | Blocking HTTP for model download |
| `dirs-next` | Cross-platform home directory |

## License

MIT
