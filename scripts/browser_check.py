"""Real-browser check of the Analysis flow (needs `pip install playwright` + `playwright install chromium`
and the app running: `streamlit run app/main.py --server.port 8501`).

    python scripts/browser_check.py [width height tag [port]]   # writes screenshots to ./browser_check/
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

w, h, tag = (int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]) if len(sys.argv) > 3 else (1366, 850, "desktop")
port = sys.argv[4] if len(sys.argv) > 4 else "8501"
out = Path(__file__).resolve().parent.parent / "browser_check"
out.mkdir(exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": w, "height": h})
    pg.goto(f"http://localhost:{port}")
    pg.wait_for_selector("text=Start Analysis", timeout=60000)
    pg.screenshot(path=str(out / f"{tag}_home.png"))
    pg.get_by_role("button", name="Try an example").click()
    pg.wait_for_selector("text=Analysis complete", timeout=180000)
    pg.wait_for_timeout(4000)
    pg.screenshot(path=str(out / f"{tag}_result.png"), full_page=True)
    b.close()
print("screenshots in", out)
