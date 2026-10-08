"""Source-bound numerical assertions for the currently selected demo controls."""

import math
import re

_CACHE = None


def close(text, expected):
    if expected is None:
        assert text == "—", (text, expected)
        return
    if isinstance(expected, str):
        assert text == expected, (text, expected)
        return
    found = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?", text, re.I)
    assert found, (text, expected)
    token = found.group().replace(",", "")
    places = len(token.split("e")[0].partition(".")[2])
    exponent = int(token.lower().partition("e")[2] or "0")
    tolerance = 0.501 * 10 ** (exponent - places)
    assert math.isclose(float(token), expected, rel_tol=1e-12, abs_tol=tolerance), (text, expected)


def cards(page, selector, expected):
    actual = page.locator(selector).all_text_contents()
    assert len(actual) == len(expected), (actual, expected)
    for text, value in zip(actual, expected, strict=True):
        close(text, value)


def image_sources(page, selector, expected):
    actual = page.locator(selector).evaluate_all('xs=>xs.map(x=>x.getAttribute("src"))')
    assert actual == expected, (actual, expected)


def check(page):
    global _CACHE
    if _CACHE is None:
        _CACHE = page.evaluate("evidence")
    if not page.locator("#preset:visible").count():
        return
    name = page.locator("#preset").input_value()
    view = page.locator("#view").input_value()
    r = next(r for r in _CACHE["rows"] if r["name"] == name)
    assert view in r["media"]
    image_sources(page, "#media", [view])
    assert page.locator("#source").get_attribute("href") == r["source"]
