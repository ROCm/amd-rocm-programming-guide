"""
Captures rocm-ontology.html and rocm-sdk-arch.html as PNG images.

Requirements:
    pip install playwright
    playwright install chromium

Usage:
    python capture-diagrams.py

Output:
    rocm-ontology.png
    rocm-sdk-arch.png
"""

from pathlib import Path
from playwright.sync_api import sync_playwright

DIR = Path(__file__).parent

DIAGRAMS = [
    {
        "html": "rocm-ontology.html",
        "png": "rocm-ontology.png",
        "selector": ".rocm-docs-ontology-diagram",
    },
    {
        "html": "rocm-sdk-arch.html",
        "png": "rocm-sdk-arch.png",
        "selector": ".rocm-docs-core-sdk-diagram",
    },
]

BOOTSTRAP_CDN = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"


def wrap_fragment(fragment: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <link rel="stylesheet" href="{BOOTSTRAP_CDN}">
  <style>
    body {{ margin: 0; padding: 16px; background: #1b1b1b; }}
  </style>
</head>
<body>
  {fragment}
</body>
</html>"""


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_viewport_size({"width": 1200, "height": 800})
        page.emulate_media(media="screen")

        for diagram in DIAGRAMS:
            html_path = DIR / diagram["html"]
            png_path = DIR / diagram["png"]

            if not html_path.exists():
                print(f"Not found: {html_path}")
                continue

            fragment = html_path.read_text(encoding="utf-8")
            page.set_content(wrap_fragment(fragment), wait_until="networkidle")

            element = page.query_selector(diagram["selector"])
            if element is None:
                print(f"Selector '{diagram['selector']}' not found in {diagram['html']}")
                continue

            element.screenshot(path=str(png_path), scale="device")
            print(f"Saved: {png_path}")

        browser.close()


if __name__ == "__main__":
    main()
