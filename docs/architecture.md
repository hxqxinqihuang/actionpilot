# Architecture

ActionPilot is organized around a small extraction pipeline:

1. Streamlit collects user input.
2. A parser normalizes the input text.
3. The orchestrator calls the task extractor.
4. The extractor uses an OpenAI-compatible provider.
5. The model response is validated with Pydantic schemas.

Future steps can add file parsers, planning, exports, calendar integration, and local persistence without changing the provider interface.
