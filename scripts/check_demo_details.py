"""Open every recorded metric disclosure and compare it with its source JSON."""

import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    root = Path(__file__).resolve().parents[1]
    html = (root / "docs/explorer.html").read_text()
    evidence = json.loads(
        re.search(r'<script[^>]*id="evidence"[^>]*>(.*?)</script>', html, re.S).group(1)
    )
    rows = evidence["rows"]
    checked = []
    errors = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1380, "height": 1024})
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.goto((root / "docs/explorer.html").as_uri())
        for row in rows:
            page.get_by_role("tab", name=row["group"], exact=True).click()
            if "task" in row:
                page.select_option("#task", row["task"])
                page.select_option("#preset", row["preset"])
            else:
                page.select_option("#preset", row["name"])
            disclosure = page.locator("details")
            if disclosure.get_attribute("open") is None:
                disclosure.locator("summary").click()
            assert disclosure.evaluate("x=>x.open")
            displayed = json.loads(page.locator("#raw").inner_text())
            assert displayed == row["metrics"], row
            source = (root / row["source"].removeprefix("../")).resolve()
            assert source.is_relative_to(root) and source.is_file()
            if row.get("kind", "").startswith("Mathematical lesson"):
                assert displayed == {}, "An illustrative lesson must not invent fitted metrics"
            else:
                original = json.loads(source.read_text())
                if row.get("kind", "").startswith("Historical"):
                    assert displayed in original["results"]
                else:
                    assert displayed == original, row["source"]
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1")
            page.set_viewport_size({"width": 1380, "height": 1024})
            disclosure.locator("summary").click()
            assert not disclosure.evaluate("x=>x.open")
            checked.append(row.get("name", row.get("preset")))
        browser.close()
    assert not errors, errors
    report = {
        "status": "passed",
        "disclosures": checked,
        "javascript_errors": errors,
        "scope": "Every recorded metric disclosure opened and closed through its summary control. Full displayed JSON equals the selected source record; illustrative lessons retain empty fitted metrics. Expanded mobile layout checked at 390 pixels.",
    }
    (root / "docs/demo_details_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Verified", len(checked), "metric disclosures against their source records")


if __name__ == "__main__":
    main()
