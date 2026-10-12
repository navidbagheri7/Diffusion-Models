"""Build an offline concept explorer from published, explicitly labeled evidence."""

import json
from pathlib import Path

from diffusion_lab.adversarial import digest
from diffusion_lab.data import GaussianMixture
from diffusion_lab.utils import save_json

GROUPS = {
    "mnist": "Image generators",
    "cifar10": "Image generators",
    "faces": "Representations and DiT",
    "imagenet": "Image generators",
    "text": "Guidance and text",
    "toy": "Flows and transport",
    "bridges": "Bridges and interpolants",
    "likelihood": "Likelihood and evaluation",
}
TEACHING = {
    "gaussian_diffusion": "Scores and reverse dynamics",
    "transport": "Flows and transport",
    "solvers": "Solvers and few-step models",
    "guidance": "Guidance and text",
    "likelihood": "Likelihood and evaluation",
    "interpolants": "Bridges and interpolants",
}


def main():
    bindings, rows = {}, []
    for metric in sorted(Path("assets/experiments").rglob("metrics.json")):
        media = sorted(p for p in metric.parent.iterdir() if p.suffix in [".png", ".gif"])
        if not media:
            continue
        group = GROUPS.get(metric.parts[2], "Image generators")
        name = metric.parent.name
        if any(
            k in name
            for k in ["cd", "ict", "consistency", "progressive", "dpm", "ddim", "edm_euler"]
        ):
            group = "Solvers and few-step models"
        if "guidance" in name or "_cfg" in name or "_classifier" in name:
            group = "Guidance and text"
        if "latent" in name or "dit" in name:
            group = "Representations and DiT"
        values = json.loads(metric.read_text())
        rows.append(
            {
                "group": group,
                "name": "/".join(metric.parent.parts[2:]),
                "kind": "Recorded trained/integration result",
                "media": ["../" + str(p) for p in media],
                "metrics": values,
                "source": "../" + str(metric),
            }
        )
        for p in [metric, *media]:
            bindings[str(p)] = digest(p)
    for folder in sorted(Path("assets/teaching").iterdir()):
        if not folder.is_dir() or folder.name == "galleries":
            continue
        media = sorted(p for p in folder.rglob("*") if p.suffix in [".png", ".gif"])
        if not media:
            continue
        group = TEACHING.get(folder.name, "Scores and reverse dynamics")
        rows.append(
            {
                "group": group,
                "name": "Teaching / " + folder.name,
                "kind": "Mathematical lesson; check analytic versus learned captions",
                "media": ["../" + str(p) for p in media],
                "metrics": {},
                "source": "../docs/23_notation_and_code.md",
            }
        )
        for p in media:
            bindings[str(p)] = digest(p)
    for metric in sorted(Path("assets/adversarial").glob("*/metrics.json")):
        folder = metric.parent
        rows.append(
            {
                "group": "GAN objectives",
                "name": folder.name,
                "kind": "New controlled synthetic GAN fit",
                "media": ["../" + str(folder / "samples.png")],
                "metrics": json.loads(metric.read_text()),
                "source": "../" + str(metric),
            }
        )
        for p in [metric, folder / "samples.png", folder / "generator.npz", folder / "samples.npz"]:
            bindings[str(p)] = digest(p)
    old = Path("assets/studies/training_seed_study/metrics.json")
    for row in json.loads(old.read_text())["results"]:
        name = f"gmm_{row['family']}_seed{row['training_seed']}"
        image = old.parent / name / "samples.png"
        rows.append(
            {
                "group": "GAN objectives",
                "name": "Historical / " + name,
                "kind": "Historical diffusion/flow reference; different sampler and evaluation seeds",
                "media": ["../" + str(image)],
                "metrics": row,
                "source": "../" + str(old),
            }
        )
        bindings[str(image)] = digest(image)
    bindings[str(old)] = digest(old)
    evidence = {"rows": rows, "means": GaussianMixture().means.tolist(), "std": 0.25}
    template = Path("scripts/templates/diffusion_explorer.html")
    script = Path("scripts/templates/diffusion_explorer.js")
    output = Path("docs/explorer.html")
    output.write_text(
        template.read_text()
        .replace("__EVIDENCE__", json.dumps(evidence, allow_nan=False).replace("<", "\\u003c"))
        .replace("__SCRIPT__", script.read_text())
    )
    for p in [
        template,
        script,
        Path("scripts") / Path(__file__).name,
        Path("diffusion_lab/paths.py"),
        Path("diffusion_lab/data.py"),
    ]:
        bindings[str(p)] = digest(p)
    save_json(
        "docs/explorer_manifest.json",
        {
            "status": "complete",
            "html_sha256": digest(output),
            "bindings": bindings,
            "presets": len(rows),
            "media": sum(len(row["media"]) for row in rows),
            "concepts": 11,
            "scope": "Live analytic Gaussian corruption, exact mixture score/Tweedie mean and prediction conversion. Other panels inspect recorded media/metrics. No network, browser training or generated intermediate checkpoints; historical weak/invalidated experiments remain visible with explicit scope.",
        },
    )
    print("Built", len(rows), "presets;", sum(len(r["media"]) for r in rows), "recorded media")


if __name__ == "__main__":
    main()
