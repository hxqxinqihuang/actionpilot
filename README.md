# ActionPilot

ActionPilot is a minimal Streamlit app that extracts structured task information from pasted notices using an OpenAI-compatible LLM API. The default provider settings target DeepSeek.

## Features in this first step

- Clear Python 3.11 project layout.
- Pydantic schemas for tasks, materials, risks, requirements, questions, and plans.
- Unified OpenAI-compatible LLM provider.
- Minimal Streamlit page for pasted text.
- Structured JSON output.
- Output language follows the dominant language of the input text.
- Missing submission deadlines are explicitly marked in the deadline output.
- API key loaded from environment variables.
- Basic error handling.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set your DeepSeek API key.

## Run

```powershell
streamlit run app.py
```

## Environment Variables

- `ACTIONPILOT_API_KEY`: Required. API key for the OpenAI-compatible provider. `DEEPSEEK_API_KEY` is also supported as a fallback.
- `ACTIONPILOT_BASE_URL`: Optional. Defaults to `https://api.deepseek.com`.
- `ACTIONPILOT_MODEL`: Optional. Defaults to `deepseek-v4-pro`.
- `ACTIONPILOT_TIMEOUT_SECONDS`: Optional. Defaults to `60`.

## Project Structure

```text
actionpilot/
+-- app.py
+-- src/
|   +-- agent/
|   +-- parsers/
|   +-- providers/
|   +-- schemas/
|   +-- storage/
|   +-- tools/
+-- examples/
+-- tests/
+-- docs/
```
