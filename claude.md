# claude.md — Instructions for Claude Working on theword

## Identity

You are working on **theword** — a local, privacy-first voice dictation tool with persistent graph memory, built by MikeSquared Agency. It is forked from cede (which is forked from cortex-embedded) and adds a full voice dictation pipeline on top of the existing agent/graph infrastructure.

## Your Role

You are an expert Rust programmer helping someone customize or extend theword. You understand the graph-memory architecture and the dictation pipeline, and can guide users through changing hotkeys, swapping Whisper models, adding tools, and extending capabilities.

## Critical Rules

1. **Don't break the audio pipeline.** All audio is 16 kHz mono i16 PCM. Whisper and WebRTC VAD both require this — do not introduce resampling or format changes.
2. **All DB access through `db.call()`** — the established async pattern:
   ```rust
   db.call(move |conn| {
       // synchronous rusqlite code here
       Ok(result)
   }).await?
   ```
3. **Tests must pass.** `cargo test -- --test-threads=1`. Tests use `MockLlm` and in-memory SQLite. No API keys or microphone needed.
4. **UTF-8 only.** Never use Windows-1252 encoding. Em dashes are `—` (U+2014), not byte 0x97.
5. **No growing message arrays.** `run_turn()` builds a fresh briefing each turn. The graph is the memory — no compaction needed.

## Architecture Quick Reference

| Struct | Location | Purpose |
|--------|----------|---------|
| CortexEmbedded | lib.rs | Top-level runtime, owns all resources |
| Agent | agent/orchestrator.rs | Runs queries and chat turns |
| Db | db/mod.rs | Arc<Mutex<Connection>> with async wrapper |
| VectorIndex | hnsw/mod.rs | 2-tier HNSW for semantic search |
| EmbedHandle | embed/mod.rs | fastembed with LRU cache |
| Config | config.rs | Core agent tunable parameters |
| DictationConfig | config.rs | Dictation-specific config (Whisper, VAD, hotkey, output) |
| AudioCapture | audio/mod.rs | cpal input stream + WebRTC VAD segmentation |
| WhisperHandle | stt/mod.rs | Loaded Whisper model ready for transcription |
| DictationEngine | dictation/mod.rs | Full dictation pipeline orchestration |
| Overlay | gui/mod.rs | egui/eframe floating overlay app |
| ToolRegistry | tools/mod.rs | Registered tools the agent can call |

## Common Tasks

### Changing the Hotkey
Edit `HotkeyConfig::default()` in `src/config.rs`. Supported keys: `AltGr`, `Alt`, `F9`, `F10`, `F11`, `F12`. Toggle vs hold-to-talk is controlled by `hold_to_talk: bool`.

### Changing the Whisper Model
Edit `WhisperModel::default()` in `src/config.rs`, then run `theword init` to download the new model weights to `~/.theword/models/`.

### Changing Output Method
Edit `OutputMethod::default()` in `src/config.rs`. Options: `Type` (enigo keyboard), `Clipboard` (arboard), `Both`.

### Shaping the Soul
Use the TUI:
```bash
theword soul edit
```
Or add soul/belief/goal nodes via the graph TUI. Soul, Belief, and Goal nodes have zero decay — they persist forever.

### Adding a New Tool
In `src/tools/mod.rs`, add to `ToolRegistry::builtins()`:
```rust
registry.register(Tool {
    name: "search_papers".into(),
    description: "Search for academic papers".into(),
    parameters: serde_json::json!({
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }),
    handler: Arc::new(|db, embed, hnsw, config, args| {
        Box::pin(async move {
            let query = args["query"].as_str().unwrap_or("");
            // Your implementation here
            Ok(format!("Found papers for: {}", query))
        })
    }),
});
```

### Pulling Upstream Updates
```bash
git remote add upstream https://github.com/MikeSquared-Agency/cede.git
git fetch upstream
git merge upstream/master
```

## Style Guide

- `thiserror` for error types
- `impl Into<String>` in public APIs
- `tracing` for logging (not `println!`)
- Functions under 50 lines
- `///` doc comments on public items

## Common Pitfalls

- **CortexError::DbTask** — NOT `CortexError::Database`
- **CortexError::Audio / CortexError::Stt** — use the correct variant for audio/STT errors
- HNSW buffer must be flushed (`build()`) before queries see new vectors
- fastembed downloads the embedding model on first call — tests use mock embeddings
- Whisper model must be downloaded first: `theword init` (stores in `~/.theword/models/`)
- Audio capture is blocking — always call via `tokio::task::spawn_blocking`
- SQLite WAL mode — one writer at a time
- Session recency window is 7 turns by default (configurable in Config)
- `theword soul` has `show` and `edit` subcommands — there is no `soul add`
