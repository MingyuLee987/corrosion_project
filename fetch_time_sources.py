"""Download openly accessible source PDFs and extract text for time auditing."""
from pathlib import Path
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader
import logging

logging.getLogger('pypdf').setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "Sherif14": "https://pdfs.semanticscholar.org/7a41/37b9f95d7bb825ead4ce6849fd23106bbf3a.pdf",
    "Eyu16": "https://pdfs.semanticscholar.org/51de/cc7fd614db5c2af4251262b83125faa44b35.pdf",
    "EH47_21": "https://pdfs.semanticscholar.org/cbc5/e58e12a9f0d3a76a55f9776e132ea62e7ead.pdf",
    "Coat24": "https://pdfs.semanticscholar.org/f446/f65aa961c15f113cf41140572aeba38258ee.pdf",
    "C1020_19": "https://www.matec-conferences.org/articles/matecconf/pdf/2019/21/matecconf_icfmce2019_01001.pdf",
    "X60_10": "https://joam.inoe.ro/articles/corrosion-studies-of-carbon-steel-x60-by-electrochemical-methods/fulltext",
}

def fetch(item):
    name, url = item
    folder = ROOT / "papers/time_verified"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / (name + ".pdf")
    try:
        if not dest.exists():
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=25) as response:
                body = response.read()
            if not body.startswith(b"%PDF"):
                raise ValueError("Response is not a PDF")
            dest.write_bytes(body)
        reader = PdfReader(dest)
        text = "\n".join(f"\n--- PDF page {i+1} ---\n{p.extract_text()}" for i, p in enumerate(reader.pages))
        dest.with_suffix(".txt").write_text(text, encoding="utf-8")
        return f"{name}: {len(reader.pages)} pages saved"
    except Exception as error:
        return f"{name}: {type(error).__name__}: {error}"

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(fetch, SOURCES.items()):
            print(result)
    for name in ("Al10_26",):
        reader = PdfReader(ROOT / "papers" / (name + ".pdf"))
        (ROOT / "papers/time_verified" / (name + ".txt")).write_text(
            "\n".join(f"\n--- PDF page {i+1} ---\n{p.extract_text()}" for i,p in enumerate(reader.pages)), encoding="utf-8")
