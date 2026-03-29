# agents.md — Guide for AI Agents Working on theword

You are working on **theword**, a local voice dictation tool with persistent graph memory. This file tells you how to navigate the codebase and contribute effectively.

## What This Repo Is

theword is forked from **cede** (the forkable agent starter kit), which is itself forked from **cortex-embedded** (the upstream engine). It adds a complete voice dictation pipeline — audio capture, VAD, Whisper STT, LLM transcript rewrite, and keyboard output — on top of the existing agent and graph-memory infrastructure.

### Ecosystem Position
- **cortex-embedded** (upstream engine) — not meant for direct forking
- **cede** — forkable agent starter kit
- **theword** (this repo) — dictation tool with graph memory

## Repository Layout

```
src/
  lib.rs              # CortexEmbedded struct, background tasks, decay, consolidation
  types.rs            # All types: Node, Edge, NodeKind, EdgeKind, Message, LlmResponse, etc.
  error.rs            # CortexError enum, Result type alias
  config.rs           # Config + DictationConfig structs with all tunable parameters
  agent/
    mod.rs            # Re-exports Agent
    orchestrator.rs   # Agent struct, run() and run_turn() methods, tool-call loop
    subagent.rs       # Sub-agent spawning and delegation
  audio/
    mod.rs            # AudioCapture — cpal input stream + WebRTC VAD segmentation
  stt/
    mod.rs            # WhisperHandle — whisper-rs wrapper; download_model(), resolve_model_path()
  dictation/
    mod.rs            # DictationEngine — full pipeline: audio → VAD → STT → LLM rewrite → output
  db/
    mod.rs            # Db struct (Arc<Mutex<Connection>>), async call() wrapper
    schema.rs         # CREATE TABLE statements, migrations
    queries.rs        # All SQL queries as functions
  embed/
    mod.rs            # EmbedHandle — fastembed wrapper with LRU cache
  hnsw/
    mod.rs            # VectorIndex — 2-tier HNSW (built index + linear buffer)
  graph/
    mod.rs            # BFS traversal, graph walk scoring
  memory/
    mod.rs            # recall(), briefing(), briefing_with_kinds(), recency window
  tools/
    mod.rs            # ToolRegistry, builtin tools (remember, recall, forget, etc.)
  llm/
    mod.rs            # LlmClient trait, AnthropicClient, OllamaClient, MockLlm
  cli/
    mod.rs            # CLI commands: Chat, Ask, Listen, Dictate, Vocab, Memory, Soul, Graph, etc.
    graph_tui.rs      # Interactive TUI graph explorer with chat panel
    graph_viz.rs      # ASCII graph visualization
  bin/
    theword.rs        # Binary entry point — calls cli::run()
tests/
  integration.rs      # Integration tests covering all phases
```

## Key Architecture

### Dictation Pipeline
`listen` (hotkey daemon) / `dictate` (one-shot) → `DictationEngine::dictate_once()`:
1. `AudioCapture::record_utterance()` — blocks until VAD detects end of speech
2. `WhisperHandle::transcribe()` — local Whisper inference (spawn_blocking)
3. LLM rewrite — filler removal, punctuation, command detection (optional)
4. `dispatch()` — enigo keyboard typing or arboard clipboard write
5. `persist_turn()` — stores `DictationTurn` node; if transcript differs significantly, stores `VocabEntry`

### Graph-Native Memory
Every node has: id (UUID), kind, content, importance, decay_rate, embedding, created_at. Nodes are connected by edges (RelatesTo, Contradicts, Supports, DerivesFrom, PartOf, Supersedes).

### Node Kinds (20)
Fact, Entity, Concept, Decision, Soul, Belief, Goal, UserInput, Session, Turn, LlmCall, ToolCall, LoopIteration, BackgroundTask, Pattern, Limitation, Capability, DictationTurn, VocabEntry, AppContext

### Chat Sessions
`run_turn(session_id, input)`: stores input as a node, builds a fresh semantic briefing, merges a recency window (last 7 turns), sends to LLM. No growing message array.

### Auto-Linking
Cosine similarity >= 0.75 adds RelatesTo edges. >= 0.85 with negation keywords triggers contradiction detection (3-tier: keyword → LLM → fallback).

### Db Pattern
All DB access: `db.call(move |conn| { ... }).await` — spawns blocking task, returns result.

## How to Customize

1. **Change the hotkey:** Edit `HotkeyConfig::default()` in `src/config.rs` (supported keys: AltGr, Alt, F9–F12)
2. **Change the Whisper model:** Edit `WhisperModel::default()` in `src/config.rs`; run `theword init` to download
3. **Change output method:** Edit `OutputMethod::default()` — `Type`, `Clipboard`, or `Both`
4. **Add tools:** Add to `ToolRegistry::builtins()` in `src/tools/mod.rs`
5. **Tune config:** See `src/config.rs` for `Config` (agent) and `DictationConfig` (dictation)
6. **Pull upstream:** `git remote add upstream <cede-url>; git fetch upstream; git merge upstream/master`

## Build and Test

```bash
cargo build
cargo test -- --test-threads=1
```

Tests use `MockLlm` and in-memory SQLite — no API keys or microphone needed.

## Conventions

- Async DB: `db.call(move |conn| { ... }).await`
- Embeddings: 384-dim f32 (BAAI/bge-small-en-v1.5)
- Node IDs: UUID v4 strings
- Timestamps: Unix seconds (i64)
- Error handling: `CortexError` enum, `Result<T>` alias
- Audio: 16 kHz mono i16 PCM (VAD and Whisper both require this)

## Branch Policy

- `master` is protected: no direct push, PRs required
- Work on `dev` branch, merge via PR
