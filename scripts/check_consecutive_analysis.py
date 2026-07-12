from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.agent.orchestrator import ExtractionOrchestrator
from src.config import AppConfig
from src.providers.factory import create_llm_provider


INPUTS = [
    "请分析任务：提交 README.md。原文没有给出具体提交日期。",
    "请分析任务：完成一个大模型 Agent 应用，并提交开源仓库链接。",
]


def main() -> None:
    config = AppConfig.from_env()
    provider = create_llm_provider(config)
    orchestrator = ExtractionOrchestrator(provider)

    for index, text in enumerate(INPUTS, start=1):
        print(f"request={index} input_chars={len(text)}")
        try:
            result = orchestrator.extract(text)
        except Exception as exc:
            print(f"request={index} status=failed error_type={type(exc).__name__} message={exc}")
        else:
            print(f"request={index} status=success tasks={len(result.tasks)}")


if __name__ == "__main__":
    main()
