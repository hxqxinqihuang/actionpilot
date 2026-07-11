from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.parsers import FileParseError, parse_uploaded_file

FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures"
FIXTURE_NAMES = [
    "course_project.txt",
    "course_project.md",
    "course_project.docx",
    "course_project.pdf",
]


def print_parse_result(file_path: Path) -> None:
    if not file_path.exists():
        print(f"[missing] {file_path}")
        return

    try:
        result = parse_uploaded_file(file_path.name, file_path.read_bytes())
    except FileParseError as exc:
        print(f"[parse error] {file_path.name}: {exc}")
        return

    print("=" * 80)
    print(f"file_name: {result.file_name}")
    print(f"file_type: {result.file_type}")
    print(f"char_count: {result.char_count}")
    print(f"page_count: {result.page_count}")
    print(f"warnings: {result.warnings}")
    print("-" * 80)
    print(result.text[:1000])
    print()


def main() -> None:
    print(f"fixtures_dir: {FIXTURES_DIR}")
    for fixture_name in FIXTURE_NAMES:
        print_parse_result(FIXTURES_DIR / fixture_name)


if __name__ == "__main__":
    main()
