# Chrome Tab Transcription System

Local-first Chrome tab transcription system foundation. This repository currently contains the Python FastAPI backend skeleton, shared protocol contracts, default configuration, and baseline tests.

Extension files, audio capture, ASR, diarization, endpointing, LLM processing, persistence, and export workflows are intentionally not implemented yet.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run The Backend

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8765
```

Health check:

```bash
curl http://127.0.0.1:8765/health
```

## Test

```bash
pytest
```

## Configuration

Defaults live in `config/default.yml`. The backend defaults to local-only binding on `127.0.0.1:8765`, local-first mode, the `balanced` scheduler profile, and no raw audio persistence.

Supported environment overrides:

- `CTTS_CONFIG_FILE`
- `CTTS_HOST`
- `CTTS_PORT`
- `CTTS_DEFAULT_PROFILE`
- `CTTS_MODE`
- `CTTS_RAW_AUDIO_PERSISTENCE`
- `CTTS_APP_VERSION`

## License

Apache License 2.0. Copyright 2026 markosiks.
