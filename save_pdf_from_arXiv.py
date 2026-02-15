import arxiv
import time
from pathlib import Path

# =========================
# 設定
# =========================
QUERY = "Neural Network"
MAX_PAPERS = 200
SAVE_DIR = Path("arxiv_pdfs")
SLEEP_SEC = 2.0   # arXivに優しく

SAVE_DIR.mkdir(exist_ok=True)

# =========================
# 検索設定
# =========================
search = arxiv.Search(
    query=QUERY,
    max_results=MAX_PAPERS,
    sort_by=arxiv.SortCriterion.Relevance,  # or SubmittedDate
)

client = arxiv.Client()

# =========================
# ダウンロード
# =========================
count = 0
for result in client.results(search):
    count += 1
    print(f"[{count}] Downloading: {result.title}")

    try:
        filename = f"{result.title[:80].replace('/', '_')}.pdf"
        result.download_pdf(dirpath=SAVE_DIR, filename=filename)

    except Exception as e:
        print(f"  -> Failed: {e}")

    time.sleep(SLEEP_SEC)

print(f"\nDone. Downloaded {count} papers.")
