"""Exercise every enabled selector tuple and slider boundaries over local HTTP.

Requires Playwright and its Chromium browser. This complements the lab-specific
numerical/browser oracle: it checks actual user events, rendered tables, byte-exact
local images, links, playback, and responsive layouts. Continuous sliders use
endpoints/midpoints; short discrete sliders use every supported position.
"""

import argparse
import hashlib
import itertools
import json
import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from demo_oracle import check as check_scientific_fields
from playwright.sync_api import sync_playwright


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", default="docs/explorer.html")
    parser.add_argument("--report", default="docs/demo_controls_validation.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    errors, assets, links, contexts = [], {}, set(), []
    slider_checks = 0
    playback_checks = 0
    try:
        with sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1380, "height": 1024})
            page.on("pageerror", lambda err: errors.append(str(err)))
            page.goto(base + "/" + args.page)
            page.wait_for_load_state("networkidle")

            def inspect():
                assert not errors, errors
                assert not re.search(
                    r"\bNaN\b|\bundefined\b|\[object Object\]", page.locator("body").inner_text()
                )
                page.wait_for_function(
                    '[...document.querySelectorAll("img")].filter(x=>x.getClientRects().length).every(x=>x.complete && x.naturalWidth>0)'
                )
                for image in page.locator("img:visible").all():
                    url = image.evaluate("x=>x.currentSrc")
                    if url.startswith(base) and url not in assets:
                        target = (root / unquote(urlparse(url).path).lstrip("/")).resolve()
                        assert target.is_relative_to(root) and target.is_file(), url
                        response = page.request.get(url)
                        assert response.ok, url
                        assert hashlib.sha256(response.body()).hexdigest() == digest(target), url
                        assets[url] = str(target.relative_to(root))
                for href in page.locator("a:visible").evaluate_all("xs=>xs.map(x=>x.href)"):
                    if href.startswith(base) and href not in links:
                        target = (root / unquote(urlparse(href).path).lstrip("/")).resolve()
                        assert target.is_relative_to(root) and target.exists(), href
                        assert page.request.head(href).ok, href
                        links.add(href)
                check_scientific_fields(page)
                for table in page.locator("table:visible").all():
                    rows = table.locator("tr:has(td)").all()
                    assert rows, "Visible table has no data rows"
                    for row in rows:
                        values = row.locator("td").all_text_contents()
                        compared = values[1:] if len(values) > 1 else values
                        assert any(v.strip() not in ["", "—", "-", "N/A"] for v in compared), values

            def controls():
                nonlocal slider_checks, playback_checks
                inputs = page.locator("input:visible:enabled")
                ranges = []
                for i in range(inputs.count()):
                    inp = inputs.nth(i)
                    if inp.get_attribute("type") != "range":
                        continue
                    lo, hi = float(inp.get_attribute("min")), float(inp.get_attribute("max"))
                    step = float(inp.get_attribute("step") or "1")
                    n = round((hi - lo) / step)
                    values = (
                        [lo + j * step for j in range(n + 1)]
                        if n <= 12
                        else [lo, lo + (n // 2) * step, hi]
                    )
                    ranges.append((i, inp.input_value(), values))
                combinations = itertools.product(*(r[2] for r in ranges))
                for values in combinations:
                    for (i, _, _), value in zip(ranges, values, strict=True):
                        inp = page.locator("input:visible:enabled").nth(i)
                        inp.evaluate(
                            '(x,v)=>{x.value=String(v);x.dispatchEvent(new Event("input",{bubbles:true}));}',
                            value,
                        )
                    inspect()
                    if ranges:
                        slider_checks += 1
                for i, original, _ in ranges:
                    page.locator("input:visible:enabled").nth(i).evaluate(
                        '(x,v)=>{x.value=v;x.dispatchEvent(new Event("input",{bubbles:true}));}',
                        original,
                    )
                play = page.locator("#play:visible")
                if play.count():
                    draw = page.locator("#draw")
                    draw.fill("1")
                    start = draw.input_value()
                    play.click()
                    page.wait_for_function('Number(document.querySelector("#draw").value)>1')
                    assert draw.input_value() != start
                    play.click()
                    stopped = draw.input_value()
                    page.wait_for_timeout(150)
                    assert draw.input_value() == stopped
                    playback_checks += 1
                inspect()

            def presets(index=0):
                selects = page.locator("select:visible:enabled")
                if index >= selects.count():
                    values = selects.evaluate_all(
                        "xs=>xs.map(x=>({field:x.id||x.dataset.control,value:x.value}))"
                    )
                    controls()
                    contexts.append({"concept": concept, "fields": values})
                    return
                values = (
                    selects.nth(index)
                    .locator("option:not([disabled])")
                    .evaluate_all("xs=>xs.map(x=>x.value)")
                )
                assert values
                for value in values:
                    page.locator("select:visible:enabled").nth(index).select_option(value)
                    presets(index + 1)

            tabs = page.get_by_role("tab").all_text_contents()
            for concept in tabs or [Path(args.page).stem]:
                if tabs:
                    page.get_by_role("tab", name=concept, exact=True).click()
                presets()
                page.set_viewport_size({"width": 390, "height": 844})
                inspect()
                assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1"), concept
                page.set_viewport_size({"width": 1380, "height": 1024})
                print("Passed", concept, len(contexts), "selector tuples", flush=True)
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    result = {
        "status": "passed",
        "page": args.page,
        "html_sha256": digest(root / args.page),
        "checker_sha256": digest(Path(__file__)),
        "selector_tuples": contexts,
        "slider_position_checks": slider_checks,
        "playback_checks": playback_checks,
        "verified_image_files": sorted(assets.values()),
        "verified_local_links": sorted(unquote(urlparse(x).path).lstrip("/") for x in links),
        "javascript_errors": errors,
        "mobile_width": 390,
        "transport": "Local HTTP, matching the user-facing web server",
        "scope": "All enabled finite selector combinations. Continuous/long sliders use endpoints and middle; discrete sliders with at most thirteen positions use every position, with joint slider combinations. Visible tables, byte-exact served images, local links, playback and mobile layouts are verified. Numerical/asset-to-model oracles are recorded by the lab-specific validators, not inferred from successful rendering.",
    }
    (root / args.report).write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Passed",
        len(contexts),
        "selector tuples,",
        slider_checks,
        "slider settings,",
        len(assets),
        "images",
        flush=True,
    )


if __name__ == "__main__":
    main()
