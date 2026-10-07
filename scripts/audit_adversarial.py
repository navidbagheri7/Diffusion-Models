"""Independently recompute all published GMM GAN metrics and portable predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diffusion_lab.adversarial import METHODS, digest, load_generator, network
from diffusion_lab.data import GaussianMixture
from diffusion_lab.metrics import gmm_metrics, mmd_squared
from diffusion_lab.utils import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-only",
        action="store_true",
        help="Verify exported evidence without private training checkpoints",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    mixture = GaussianMixture()
    checked = []
    for method in METHODS:
        for seed in [42, 43, 44]:
            folder = Path("assets/adversarial") / f"{method}_{seed}"
            metric = json.loads((folder / "metrics.json").read_text())
            for filename, value in metric["source_sha256"].items():
                assert digest(Path("diffusion_lab") / filename) == value
            assert digest(folder / "samples.npz") == metric["samples_sha256"]
            assert digest(folder / "generator.npz") == metric["generator_sha256"]
            model = load_generator(folder)
            with np.load(folder / "samples.npz", allow_pickle=False) as data:
                latent, samples, reference = [
                    torch.from_numpy(data[key].copy())
                    for key in ["latent", "generated", "reference"]
                ]
            with torch.no_grad():
                torch.testing.assert_close(model(latent), samples, rtol=0, atol=0)
            np.testing.assert_array_equal(
                latent.numpy(),
                torch.randn(4096, 2, generator=torch.Generator().manual_seed(300000)).numpy(),
            )
            np.testing.assert_array_equal(
                reference.numpy(),
                mixture.sample(4096, torch.Generator().manual_seed(400000))[0].numpy(),
            )
            recomputed = gmm_metrics(samples, reference, mixture)
            for key, value in recomputed.items():
                np.testing.assert_allclose(value, metric[key], rtol=1e-7, atol=1e-8)
            with torch.no_grad():
                val = model(torch.randn(1024, 2, generator=torch.Generator().manual_seed(100000)))
            score = mmd_squared(val, mixture.sample(1024, torch.Generator().manual_seed(200000))[0])
            np.testing.assert_allclose(score, metric["validation_mmd"], rtol=1e-7, atol=1e-8)
            history = json.loads((folder / "history.json").read_text())
            assert metric["config"]["steps"] == metric["generator_updates"] == 6000
            assert [r["step"] for r in history] == list(range(500, 6001, 500))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                initial = network()
                with torch.no_grad():
                    initial_score = mmd_squared(
                        initial(
                            torch.randn(1024, 2, generator=torch.Generator().manual_seed(100000))
                        ),
                        mixture.sample(1024, torch.Generator().manual_seed(200000))[0],
                    )
            candidates = [{"step": 0, "validation_mmd": initial_score}, *history]
            selected = min(candidates, key=lambda row: row["validation_mmd"])
            assert selected["step"] == metric["selected_step"]
            np.testing.assert_allclose(
                selected["validation_mmd"], metric["validation_mmd"], rtol=1e-7, atol=1e-8
            )
            assert metric["selected_step"] == 0 or any(
                r["step"] == metric["selected_step"]
                and r["validation_mmd"] == metric["validation_mmd"]
                for r in history
            )
            if not args.public_only:
                state = torch.load(
                    Path("outputs/adversarial") / folder.name / "checkpoint.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                assert state["step"] == 6000 and state["best_step"] == metric["selected_step"]
                for key, array in model.state_dict().items():
                    torch.testing.assert_close(array, state["best_weights"][key], rtol=0, atol=0)
            checked.append(
                {
                    "method": method,
                    "seed": seed,
                    "selected_step": metric["selected_step"],
                    "generator_sha256": metric["generator_sha256"],
                    "full_test_predictions_reproduced": 4096,
                    "metrics_recomputed": True,
                }
            )
            print("Verified", folder, flush=True)
    save_json(
        "assets/reports/adversarial_validation.json",
        {
            "status": "passed",
            "fits": checked,
            "heldout_generation_count": 4096,
            "reference_count": 4096,
            "private_checkpoint_identity_checked": not args.public_only,
            "scope": "All fifteen fitted generators, validation selection values and independent recomputation of every public test metric. Private best-state identities are checked only without --public-only. Original diffusion/flow evidence remains historical and has a different source/seed/compute protocol.",
        },
    )


if __name__ == "__main__":
    main()
