"""Every gallery selector plus independent path/score calculations in Chromium."""

import json
import re
from pathlib import Path

import numpy as np
import torch
from playwright.sync_api import sync_playwright

from diffusion_lab.adversarial import digest
from diffusion_lab.data import GaussianMixture
from diffusion_lab.paths import GaussianPath
from diffusion_lab.utils import save_json

FIELDS = {
    "nfe": ["nfe_per_sample", "nfe_per_image"],
    "calls": ["field_calls", "network_calls"],
    "probes": ["probe_count", "probes"],
    "log_density": ["mean_log_density_nats"],
    "density_bpd": ["dequantized_density_bpd_estimate"],
    "elbo_bpd": ["negative_elbo_bpd_mean"],
    "density_rmse": ["rmse_log_density"],
    "density_mae": ["mean_absolute_error"],
    "mmd": ["mmd_squared_unbiased"],
    "w2": ["sliced_w2"],
    "fid": ["fid_2048", "fid", "frechet_inception_distance"],
    "kid": ["kid_mean"],
    "digit": ["frechet_digit_features_128"],
    "accuracy": ["requested_class_accuracy"],
    "clip": ["clip_cosine_mean"],
    "clip_margin": ["paired_minus_mismatched_mean"],
    "coverage": ["mode_coverage_3sigma"],
    "samples": ["generated_count", "sample_count", "samples"],
    "reference": ["real_count", "reference_count"],
}
PRIMARY = set(FIELDS) - {"nfe", "calls", "probes", "samples", "reference"}


def records(value):
    if isinstance(value, list):
        return value
    if isinstance(value.get("results"), list):
        return [value | row for row in value["results"]]
    return [value]


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def displayed(text, value):
    if isinstance(value, bool):
        assert text == str(value).lower(), (text, value)
    elif numeric(value):
        # Browser formatting is four decimals or three exponential decimals.
        tolerance = abs(value) * 5.01e-4 if "e" in text.lower() else 5.01e-5
        assert abs(float(text) - value) <= tolerance, (text, value)
    else:
        assert text == str(value), (text, value)


def main():
    errors, checked, live, mobile = [], [], [], []
    details_checked = 0
    comparisons_checked = []
    evidence = json.loads(
        re.search(
            r'<script id="evidence" type="application/json">(.*?)</script>',
            Path("docs/explorer.html").read_text(),
            re.S,
        ).group(1)
    )
    by_name = {row["name"]: row for row in evidence["rows"]}
    for row in evidence["rows"]:
        if row["kind"].startswith("Mathematical lesson"):
            assert row["metrics"] == {}
            continue
        source = Path(row["source"].removeprefix("../"))
        original = json.loads(source.read_text())
        if row["kind"].startswith("Historical"):
            assert row["metrics"] in original["results"]
            assert row["name"] == (
                "Historical / gmm_"
                + row["metrics"]["family"]
                + "_seed"
                + str(row["metrics"]["training_seed"])
            )
        else:
            assert row["metrics"] == original, row["name"]
            assert all(
                Path(view.removeprefix("../")).parent == source.parent for view in row["media"]
            )
    mixture = GaussianMixture()
    manifest = json.loads(Path("docs/explorer_manifest.json").read_text())
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1380, "height": 1024})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(Path("docs/explorer.html").resolve().as_uri())
        # Regression for the reported blank likelihood table, independently
        # anchored to a source report rather than the generated page payload.
        page.get_by_role("tab", name="Likelihood and evaluation", exact=True).click()
        likelihood = json.loads(
            Path("assets/experiments/likelihood/likelihood_cifar10_edm/metrics.json").read_text()
        )
        assert (
            f"{likelihood['dequantized_density_bpd_estimate']:.4f}"
            in page.locator("#comparison").inner_text()
        ), "Likelihood comparison omits the measured bits/dimension estimate"
        page.get_by_role("tab", name="Forward noising", exact=True).click()
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
                comparison_rows = page.locator("#comparison tr[data-source-name]")
                actual_names = set()
                for element in comparison_rows.all():
                    name = element.get_attribute("data-source-name")
                    index = int(element.get_attribute("data-record-index"))
                    original = records(by_name[name]["metrics"])[index]
                    actual_names.add((name, index))
                    cells = {
                        cell.get_attribute("data-metric"): cell.inner_text()
                        for cell in element.locator("td[data-metric]").all()
                    }
                    assert any(cells.get(key, "—") != "—" for key in PRIMARY)
                    for key, aliases in FIELDS.items():
                        value = next(
                            (original[k] for k in aliases if numeric(original.get(k))), None
                        )
                        if value is not None:
                            assert key in cells, (name, key)
                            displayed(cells[key], value)
                    comparisons_checked.append({"group": group, "name": name, "result": index})
                expected_names = {
                    (row["name"], index)
                    for row in evidence["rows"]
                    if row["group"] == group
                    for index, record in enumerate(records(row["metrics"]))
                    if any(numeric(record.get(key)) for field in PRIMARY for key in FIELDS[field])
                }
                assert actual_names == expected_names, (group, expected_names - actual_names)
                for table in page.locator("#comparison table").all():
                    assert table.locator("tr[data-source-name]").count() > 0
                    for key in (
                        table.locator("tr[data-source-name]")
                        .first.locator("td[data-metric]")
                        .evaluate_all("xs=>xs.map(x=>x.dataset.metric)")
                    ):
                        assert any(
                            text != "—"
                            for text in table.locator(
                                f'td[data-metric="{key}"]'
                            ).all_text_contents()
                        ), (group, key)
                for name in page.locator("#preset option").all_text_contents():
                    page.select_option("#preset", name)
                    row = by_name[name]
                    expected_records = records(row["metrics"])
                    shown_fields = set()
                    for entry in page.locator("#metrics tr[data-field]").all():
                        index = int(entry.get_attribute("data-record-index"))
                        key = entry.get_attribute("data-field")
                        displayed(entry.locator("td").inner_text(), expected_records[index][key])
                        shown_fields.add((index, key))
                        details_checked += 1
                    assert {
                        (index, key)
                        for index, record in enumerate(expected_records)
                        for key, value in record.items()
                        if numeric(value) or isinstance(value, bool)
                    } <= shown_fields, name
                    assert page.locator("#metrics tr").count() > 0
                    for view in page.locator("#view option").evaluate_all(
                        "(options)=>options.map(x=>x.value)"
                    ):
                        page.select_option("#view", view)
                        page.wait_for_function(
                            'document.querySelector("#media").complete && document.querySelector("#media").naturalWidth>0'
                        )
                        assert page.locator("#title").inner_text() == name
                        assert page.locator("#media").get_attribute("src") == view
                        assert view in row["media"]
                        assert page.locator("#source").get_attribute("href") == row["source"]
                        assert not re.search(
                            r"\b(?:NaN|undefined)\b|\[object Object\]",
                            page.locator("#recorded").inner_text(),
                        )
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
        page.get_by_role("tab", name="Likelihood and evaluation", exact=True).click()
        page.locator("#comparison").screenshot(path="assets/previews/likelihood_comparison.png")
        page.get_by_role("tab", name="Representations and DiT", exact=True).click()
        page.locator("#comparison").screenshot(path="assets/previews/image_metrics_comparison.png")
        page.get_by_role("tab", name="GAN objectives", exact=True).click()
        page.select_option("#preset", "wasserstein_gp_42")
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
            "comparison_records": comparisons_checked,
            "checked_metric_fields": details_checked,
            "comparison_scope": "Every applicable scalar comparison is checked against the original embedded report; every selected detail field and image/source selector is verified. Empty lessons explicitly declare the absence of fitted comparisons. Likelihood bounds, density estimates, image features and caption alignment have separate tables.",
        },
    )
    print(
        "Passed", len(checked), "gallery views and", len(live), "independent analytic calculations"
    )


if __name__ == "__main__":
    main()
