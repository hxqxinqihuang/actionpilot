from __future__ import annotations

import json

import streamlit as st

from src.agent.orchestrator import ExtractionOrchestrator
from src.config import AppConfig, ConfigError
from src.parsers.text_parser import parse_text
from src.providers.factory import create_llm_provider
from src.providers.openai_compatible import LLMProviderError


st.set_page_config(page_title="ActionPilot", page_icon="AP", layout="wide")

st.title("ActionPilot")
st.caption("Paste a notice or requirement text to extract actionable task details.")


def build_orchestrator() -> ExtractionOrchestrator:
    config = AppConfig.from_env()
    provider = create_llm_provider(config)
    return ExtractionOrchestrator(provider=provider)


notice_text = st.text_area(
    "Notification text",
    height=280,
    placeholder="Paste course notices, competition announcements, job descriptions, or other task-related text here.",
)

submitted = st.button("Extract JSON", type="primary")

if submitted:
    cleaned_text = parse_text(notice_text)
    if not cleaned_text:
        st.warning("Please paste some text before extracting.")
    else:
        try:
            orchestrator = build_orchestrator()
            with st.spinner("Extracting structured task information..."):
                result = orchestrator.extract(cleaned_text)
            st.success("Extraction complete.")
            st.json(json.loads(result.model_dump_json()))
        except ConfigError as exc:
            st.error(str(exc))
            st.info("Create a .env file or export environment variables based on .env.example.")
        except LLMProviderError as exc:
            st.error(str(exc))
            st.info("Check your DeepSeek API key, base URL, model name, and account availability.")
        except Exception as exc:
            st.error("Extraction failed. Please check your model settings and try again.")
            st.caption(f"Details: {exc}")
