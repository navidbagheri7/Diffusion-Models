"""Every gallery selector plus independent path/score calculations in Chromium."""

import json
from pathlib import Path

import numpy as np
import torch
from playwright.sync_api import sync_playwright

from diffusion_lab.adversarial import digest
from diffusion_lab.data import GaussianMixture
from diffusion_lab.paths import GaussianPath
from diffusion_lab.utils import save_json


def main():
    errors, checked, live, mobile = [], [], [], []
    mixture = GaussianMixture()
    manifest = json.loads(Path("docs/explorer_manifest.json").read_text())
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1380, "height": 1024})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(Path("docs/explorer.html").resolve().as_uri())
        for kind in ["vp_linear", "vp_cosine", "sub_vp", "ve", "vp_logsnr", "vp_exponential"]:
            for time in [0.001, 0.1, 0.25, 0.5, 0.9, 0.999]:
                page.select_option("#path", kind)
                page.locator("#time").fill(str(time))
                actual = page.evaluate("window.liveDiff")
                t = torch.tensor([time], dtype=torch.float64)
                a, s = GaussianPath(kind).coefficients(t)
                x = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
                score = mixture.score(x, a, s)
                posterior = (x + s[:, None] ** 2 * score) / a[:, None]
                np.testing.assert_allclose(
                    [actual["alpha"], actual["sigma"]], [float(a), float(s)], rtol=1e-12, atol=1e-12
                )
                np.testing.assert_allclose(actual["score"], score[0], rtol=2e-10, atol=2e-10)
                np.testing.assert_allclose(actual["posterior"], posterior[0], rtol=2e-7, atol=1e-7)
                np.testing.assert_allclose(
                    actual["logDensity"], mixture.log_prob(x, a, s)[0], rtol=2e-10, atol=2e-10
                )
                live.append({"path": kind, "time": time})
        page.select_option("#path", "vp_linear")
        page.locator("#time").fill("0.25")
        page.screenshot(path="assets/previews/noising_desktop.png", full_page=True)
        groups = page.locator("nav button").all_text_contents()
        for group in groups:
            page.get_by_role("tab", name=group, exact=True).click()
            if group not in ["Forward noising", "Prediction targets"]:
                for name in page.locator("#preset option").all_text_contents():
                    page.select_option("#preset", name)
                    for view in page.locator("#view option").evaluate_all(
                        "(options)=>options.map(x=>x.value)"
                    ):
                        page.select_option("#view", view)
                        page.wait_for_function(
                            'document.querySelector("#media").complete && document.querySelector("#media").naturalWidth>0'
                        )
                        assert page.locator("#title").inner_text() == name
                        checked.append({"group": group, "name": name, "view": view})
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth<=innerWidth"), group
            mobile.append(group)
            page.set_viewport_size({"width": 1380, "height": 1024})
        page.get_by_role("tab", name="GAN objectives", exact=True).click()
        page.select_option("#preset", "wasserstein_gp_42")
        page.wait_for_function(
            'document.querySelector("#media").complete && document.querySelector("#media").naturalWidth>0'
        )
        page.screenshot(path="assets/previews/gan_desktop.png", full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path="assets/previews/explorer_mobile.png", full_page=True)
        browser.close()
    assert not errors, errors
    assert len(checked) == manifest["media"]
    save_json(
        "docs/explorer_browser_validation.json",
        {
            "status": "passed",
            "html_sha256": digest("docs/explorer.html"),
            "checked_views": checked,
            "live_analytic_checks": live,
            "mobile_concepts": mobile,
            "javascript_errors": errors,
        },
    )
    print(
        "Passed", len(checked), "gallery views and", len(live), "independent analytic calculations"
    )


if __name__ == "__main__":
    main()
