"""Extract and render all Mermaid diagrams from the CW-019 architecture document."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENT = ROOT / "docs" / "architecture.md"
DEFAULT_OUTPUT = ROOT / "artifacts" / "reports" / "cw019_architecture_render"
MERMAID_FENCE = re.compile(r"\x60\x60\x60mermaid\r?\n(.*?)\r?\n\x60\x60\x60", re.DOTALL)


def render(document: Path, output_dir: Path, npx: str, package: str) -> dict[str, object]:
    text = document.read_text(encoding="utf-8")
    blocks = MERMAID_FENCE.findall(text)
    if not blocks:
        raise ValueError("no_mermaid_diagrams")
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[str] = []
    for index, block in enumerate(blocks, start=1):
        source = output_dir / f"diagram-{index:02d}.mmd"
        image = output_dir / f"diagram-{index:02d}.png"
        source.write_text(block + "\n", encoding="utf-8")
        command = [npx, "-y", package, "-i", str(source), "-o", str(image)]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0 or not image.is_file():
            raise RuntimeError(f"diagram_{index}_render_failed")
        rendered.append(image.as_posix())
    return {"document": document.as_posix(), "diagram_count": len(blocks), "rendered": rendered}


def main() -> int:
    parser = argparse.ArgumentParser(description="Render CW-019 Mermaid diagrams with a pinned CLI package.")
    parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--npx", default="npx", help="npx executable, e.g. npx.cmd on Windows")
    parser.add_argument("--package", default="@mermaid-js/mermaid-cli@12.0.0")
    args = parser.parse_args()
    try:
        result = render(args.document, args.output_dir, args.npx, args.package)
    except (OSError, RuntimeError, ValueError):
        print(json.dumps({"ok": False, "error": "architecture_render_failed"}, ensure_ascii=False))
        return 1
    result["ok"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
