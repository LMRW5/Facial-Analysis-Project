from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._skip_depth = 0
        self._block_tags = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "li", "tr", "br"}

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag in self._block_tags and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data.strip())

    def get_text(self) -> str:
        return "\n".join(part for part in self._parts if part).strip()


def fetch_page_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=20) as response:
        html = response.read().decode("utf-8", errors="replace")

    parser = TextExtractor()
    parser.feed(html)
    text = parser.get_text()
    return text or "No description found on the target page."


def build_candidate_names(phenotype_name: str) -> List[str]:
    base = phenotype_name.strip()
    if not base:
        return []
    candidates = [base]
    if base.islower():
        candidates.append(base.capitalize())
    if len(base) > 1:
        candidates.append(base[0].upper() + base[1:])
    return list(dict.fromkeys(candidates))


def scrape_phenotypes(dataset_dir: Path) -> None:
    phenotype_dirs = [path for path in dataset_dir.iterdir() if path.is_dir()]
    if not phenotype_dirs:
        print(f"No phenotype folders found in {dataset_dir}")
        return

    for phenotype_dir in sorted(phenotype_dirs):
        phenotype_name = phenotype_dir.name
        description_path = phenotype_dir / "webData.txt"

        for candidate_name in build_candidate_names(phenotype_name):
            url = f"http://humanphenotypes.net/{candidate_name}.html"
            try:
                text = fetch_page_text(url)
            except (HTTPError, URLError, TimeoutError, ValueError):
                continue

            description_path.write_text(text, encoding="utf-8")
            print(f"Saved description for {phenotype_name}: {url}")
            break
        else:
            fallback_text = f"Could not fetch description for {phenotype_name}."
            description_path.write_text(fallback_text, encoding="utf-8")
            print(f"Could not fetch description for {phenotype_name}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset"
    scrape_phenotypes(dataset_dir)
    print("Scraping complete.")
