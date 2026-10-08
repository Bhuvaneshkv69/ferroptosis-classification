"""Real-browser acceptance run of the result popup, latest-analysis card, dark mode and responsive layout.
Needs the app running (`streamlit run app/main.py`) and Playwright + Chromium. Prints PASS/FAIL per check, exit code 1 on any FAIL.

    python scripts/ui_acceptance.py <input_dir> [out_dir] [port]
        input_dir must hold core_upload.csv (raw-scale expression of the 35 genes; see docs/CSV_FORMAT.md).
"""
import re
import sys
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
IN = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else IN
PORT = sys.argv[3] if len(sys.argv) > 3 else "8501"
URL = f"http://localhost:{PORT}"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def bg(pg):
    return pg.evaluate("getComputedStyle(document.querySelector('[data-testid=stApp]')).backgroundColor")


def no_exc(pg):
    return not pg.query_selector("[data-testid=stException]")


def fresh(b, theme=None, w=1280, h=900):
    pg = b.new_page(viewport={"width": w, "height": h})
    pg.goto(URL + (f"/?theme={theme}" if theme else ""))
    pg.wait_for_selector("text=Start Analysis", timeout=60000)
    return pg


def upload_and_run(pg, csv, model_label=None):
    pg.get_by_role("button", name="Start Analysis").click()
    pg.wait_for_selector("input[type=file]", state="attached", timeout=30000)
    pg.set_input_files("input[type=file]", str(IN / csv))
    pg.wait_for_timeout(2500)
    pg.get_by_role("button", name="Continue").first.click()
    pg.wait_for_timeout(2500)
    if model_label:
        pg.get_by_text(model_label, exact=True).first.click()
        pg.wait_for_timeout(1000)
    pg.get_by_text("Classification", exact=True).first.click()
    pg.wait_for_timeout(800)
    pg.get_by_role("button", name="Run analysis").click()


def real_first_sample_output(csv):
    import model_registry as reg
    import predict as prd
    b = reg.load_model(reg.current_best("classification"))
    d = pd.read_csv(IN / csv)
    out = prd.classify(b, prd.to_log2(d, b["features"], False))
    mt = prd.member_table(b, prd.to_log2(d, b["features"], False).iloc[[0]])
    return out.iloc[0], mt


with sync_playwright() as p:
    b = p.chromium.launch()

    # ---------------- A. light mode + C. CSV flow + F-I popup
    pg = fresh(b)
    check("A light mode loads", bg(pg) == "rgb(252, 252, 251)" and no_exc(pg), bg(pg))
    pg.screenshot(path=str(OUT / "acc_home_light.png"))
    upload_and_run(pg, "core_upload.csv")
    pg.wait_for_selector("[role=dialog]", timeout=240000)
    pg.wait_for_timeout(2500)
    dlg = pg.inner_text("[role=dialog]")
    real, mt = real_first_sample_output("core_upload.csv")
    expect_cls = "MSI-H" if real["predicted_class"] == "MSI-H" else "MSS / MSI-L"
    check("C/F popup appears automatically after CSV analysis", "Analysis complete" in dlg and "MODEL RESULT" in dlg.upper())
    check("popup shows the model's real predicted class", f"Predicted: {expect_cls}" in dlg, expect_cls)
    shown = re.search(r"(\d+)%\s*\n", dlg)
    check("G popup shows the model's real probability (not a constant)", shown and int(shown.group(1)) == round(100 * float(real["model_probability"])),
          f"shown {shown.group(1) if shown else None}% vs model {100 * float(real['model_probability']):.1f}%")
    pos = "MSI-H"
    votes = [("MSI-H" if s else "MSS/MSI-L") for s in mt["supports_high"]]
    agree = sum(v == real["predicted_class"] for v in votes)
    check("H popup model agreement equals the real member predictions", f"{agree} of {len(votes)} models agree" in dlg, f"{agree} of {len(votes)}")
    check("popup explains the result in plain words and carries the research disclaimer",
          "What does this mean?" in dlg and "not be interpreted as a medical diagnosis" in dlg)
    check("popup keeps sample prediction and model performance apart",
          "MODEL PERFORMANCE" in dlg.upper() and "Locked-test accuracy" in dlg and "Test AUC" in dlg)
    low = dlg.lower()
    check("neutral language (no success rate / healthy / danger / positive-negative)",
          not any(w in low for w in ("success", "healthy", "danger", "positive", "negative", "diagnosis confirmed")))
    pg.get_by_text("How was this result produced?").click()
    pg.wait_for_timeout(800)
    inner = pg.inner_text("[role=dialog]")
    check("expandable 'How was this result produced?' lists the real models",
          all(k in inner for k in ("Logistic regression", "Random forest", "Extra-trees", "support-vector", "XGBoost")) and "Combined result" in inner)
    pg.screenshot(path=str(OUT / "acc_popup_light.png"))
    # tooltip text
    pg.locator("[role=dialog] [data-testid=stTooltipIcon]").first.hover()
    pg.wait_for_timeout(700)
    check("tooltip explains probability vs accuracy", "not the same thing" in pg.inner_text("body"))
    # keyboard: Escape closes
    pg.keyboard.press("Escape")
    pg.wait_for_timeout(1500)
    check("Escape closes the popup", pg.locator("[role=dialog]").count() == 0)
    check("I detailed results stay available after closing (performance card present)",
          "PRODUCTION MODEL AND ITS PERFORMANCE" in pg.inner_text("body").upper() and "Validation accuracy" in pg.inner_text("body"))
    # ---------------- J. latest analysis
    pg.get_by_role("button", name="New analysis").click()
    pg.wait_for_timeout(2500)
    body = pg.inner_text("body")
    check("J latest analysis card is visible at the top of the workspace after closing",
          "Latest analysis" in body and "CSV upload" in body and f"Predicted {expect_cls}" in body)
    check("latest analysis shows no sample identifier", "TCGA-" not in body.split("Latest analysis")[1][:400])
    pg.screenshot(path=str(OUT / "acc_latest_light.png"))
    pg.get_by_role("button", name="Open detailed results").click()
    pg.wait_for_timeout(5000)
    check("clicking it reopens the detailed result", "Analysis complete" in pg.inner_text("body") and "Predicted subgroup" in pg.inner_text("body"))
    # keyboard focus on Close
    pg.get_by_role("button", name="New analysis").click()
    pg.wait_for_timeout(2000)
    pg.get_by_role("button", name="Show summary").click()
    pg.wait_for_selector("[role=dialog]", timeout=60000)
    pg.wait_for_timeout(1500)
    ring = None
    for _ in range(25):                         # real keyboard navigation (programmatic focus does not trigger :focus-visible)
        pg.keyboard.press("Tab")
        if pg.evaluate("document.activeElement.textContent.trim()") == "Close":
            ring = pg.evaluate("(()=>{const e=document.activeElement;const cs=getComputedStyle(e);return [e.textContent.trim(), cs.outlineStyle, cs.outlineWidth]})()")
            break
    check("keyboard Tab reaches Close and shows a visible focus ring", ring and ring[1] != "none" and ring[2] != "0px", str(ring))
    pg.keyboard.press("Enter")
    try:
        pg.wait_for_selector("[role=dialog]", state="detached", timeout=20000)
        closed = True
    except Exception:
        closed = False
    check("Close button (keyboard Enter) closes the popup", closed)
    pg.close()

    # ---------------- B. dark mode + persistence + popup in dark
    pg = fresh(b)
    pg.get_by_text("Dark", exact=False).first.click()
    pg.wait_for_timeout(3500)
    check("B dark mode loads (not pure black)", bg(pg) == "rgb(11, 17, 32)" and no_exc(pg), bg(pg))
    pg.reload()
    pg.wait_for_selector("[data-testid=stApp]", timeout=60000)
    pg.wait_for_timeout(4000)
    check("dark mode persists across a browser refresh", bg(pg) == "rgb(11, 17, 32)")
    pg.goto(URL + "/")
    pg.wait_for_timeout(6500)
    check("dark mode is restored from browser storage on a plain URL", bg(pg) == "rgb(11, 17, 32)")
    upload_and_run(pg, "core_upload.csv")
    pg.wait_for_selector("[role=dialog]", timeout=240000)
    pg.wait_for_timeout(2500)
    card = pg.evaluate("(()=>{const e=document.querySelector('[data-testid=stDialog] > div');const cs=getComputedStyle(e);return [cs.backgroundColor, Math.round(e.getBoundingClientRect().width)]})()")
    check("popup is dark in dark mode and within the viewport", card[0] != "rgb(252, 252, 251)" and card[1] <= 1280, str(card))
    pg.screenshot(path=str(OUT / "acc_popup_dark.png"))
    pg.keyboard.press("Escape")
    pg.wait_for_timeout(1500)
    # L. navigation in dark: every page, no exception, charts not white
    for title in ("Model Lab", "Single-model details", "Data", "Train / Compare", "Documentation", "Analysis"):
        vis = pg.locator("[data-testid=stSidebarNav]").count() and pg.locator("[data-testid=stSidebarNav]").first.is_visible()
        if not vis:
            pg.locator("[data-testid=stExpandSidebarButton], [data-testid=stSidebarCollapsedControl] button").first.click(timeout=4000)
            pg.wait_for_timeout(500)
        pg.locator("[data-testid=stSidebarNavLink]", has_text=title).first.click()
        pg.wait_for_timeout(6500)
        white_chart = pg.evaluate("""(()=>{const c=document.querySelector('.js-plotly-plot .main-svg');return c?getComputedStyle(c).backgroundColor:null})()""")
        check(f"L dark navigation: {title}", no_exc(pg) and bg(pg) == "rgb(11, 17, 32)" and white_chart in (None, "rgba(0, 0, 0, 0)"),
              f"chart bg {white_chart}")
        if title == "Data":
            pg.screenshot(path=str(OUT / "acc_data_dark.png"))
    pg.close()

    # ---------------- manual entry (D) in dark
    pg = fresh(b, "dark")
    pg.get_by_role("button", name="Start Analysis").click()
    pg.wait_for_timeout(1500)
    pg.get_by_text("Enter sample", exact=True).first.click()
    pg.wait_for_timeout(1500)
    pg.get_by_role("button", name="Continue").first.click()
    pg.wait_for_timeout(2500)
    pg.get_by_text("Classification", exact=True).first.click()
    pg.wait_for_timeout(800)
    pg.get_by_role("button", name="Run analysis").click()
    pg.wait_for_selector("[role=dialog]", timeout=240000)
    pg.wait_for_timeout(2000)
    mdlg = pg.inner_text("[role=dialog]")
    check("D manual entry shows the same popup", "MODEL RESULT" in mdlg.upper() and "Predicted:" in mdlg and "Prediction probability" in mdlg)
    pg.get_by_role("button", name="View Detailed Results").click()
    pg.wait_for_timeout(6000)
    check("manual entry: View Detailed Results closes the popup and shows details",
          pg.locator("[role=dialog]").count() == 0 and "Predicted subgroup" in pg.inner_text("body"))
    pg.close()

    # ---------------- extended model label
    pg = fresh(b)
    upload_and_run(pg, "ext_upload.csv", model_label="35-gene + MMR Extended Model (39 genes)")
    pg.wait_for_selector("[role=dialog]", timeout=240000)
    pg.wait_for_timeout(2000)
    edlg = pg.inner_text("[role=dialog]")
    check("extended model is labelled as such and not as a ferroptosis classifier",
          "35-gene + MMR Extended Model (39 genes)" in edlg and "not a ferroptosis-only classifier" in edlg and "Ferroptosis Core Classifier" not in edlg)
    check("single extended model shows no invented agreement", "no model agreement" in edlg and "models agree" not in edlg)
    pg.close()

    # ---------------- responsive: mobile and tablet
    for w, h, name in ((390, 844, "mobile"), (820, 1180, "tablet"), (1440, 900, "desktop")):
        pg = fresh(b, None, w, h)
        upload_and_run(pg, "core_upload.csv")
        pg.wait_for_selector("[role=dialog]", timeout=240000)
        pg.wait_for_timeout(2500)
        m = pg.evaluate("""(()=>{const e=document.querySelector('[data-testid=stDialog] > div').getBoundingClientRect();
            return [Math.round(e.left), Math.round(e.right), window.innerWidth, document.documentElement.scrollWidth]})()""")
        check(f"popup fits the {name} viewport ({w}px)", m[0] >= 0 and m[1] <= m[2] and m[3] <= m[2] + 1, str(m))
        btn = pg.get_by_role("button", name="View Detailed Results")
        btn.scroll_into_view_if_needed()
        check(f"{name}: actions are reachable", btn.is_visible() and pg.locator("[role=dialog] button:has-text('Close')").is_visible())
        pg.screenshot(path=str(OUT / f"acc_popup_{name}.png"))
        pg.close()

    # ---------------- K. error handling: a file that cannot be analysed never produces a result
    pg = fresh(b)
    pg.get_by_role("button", name="Start Analysis").click()
    pg.wait_for_selector("input[type=file]", state="attached", timeout=30000)
    pg.set_input_files("input[type=file]", str(IN / "bad_missing_genes.csv"))
    pg.wait_for_timeout(3000)
    body = pg.inner_text("body")
    check("K invalid input is explained in plain words and no result appears",
          ("missing" in body.lower() or "required" in body.lower()) and pg.locator("[role=dialog]").count() == 0 and "Predicted:" not in body)
    cont = pg.get_by_role("button", name="Continue")
    check("K analysis cannot be started from an invalid file", cont.count() == 0 or not cont.first.is_enabled())
    pg.close()
    b.close()

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
