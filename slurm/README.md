# Cluster runs

Templates are provided; no Slurm jobs are submitted automatically. Activate
your environment before submission, or export PYTHON_BIN to its executable.
From the repository root, create the log directory first:

~~~bash
mkdir -p slurm/logs
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT slurm/train_images.sbatch
sbatch slurm/train_single.sbatch data=mnist model.channels=1 model.out_channels=1
sbatch slurm/bridge.sbatch --initialization imf --iterations 4
sbatch slurm/train_dit.sbatch
VAE_PATH=/path/to/vae sbatch slurm/cache_latents.sbatch
RECIPE=image_latent_dit sbatch slurm/train_dit.sbatch
TEXT_BASE_MODEL=/path/to/local/sd2 sbatch slurm/train_text.sbatch
TEXT_BASE_MODEL=/path/to/local/sd2 sbatch slurm/train_coco_scratch.sbatch
sbatch slurm/train_cifar_consistency.sbatch teacher=/path/to/frozen/edm.pt
STUDY=consistency sbatch slurm/evidence_study.sbatch
~~~

Adjust GPU requests, account, partition, memory and wall time for your site.
The three-GPU template runs one DDP job; batch_size is per GPU and accumulation
multiplies the effective global batch. Slurm assigns visible GPUs, so do not
override CUDA_VISIBLE_DEVICES. For multiple nodes, configure rendezvous using
your site's launcher; the included template is single-node.

Resume with the same DDP world size and configuration:

~~~bash
sbatch slurm/train_images.sbatch resume=outputs/cifar10_edm_slurm/checkpoint.pt
~~~

Checkpoints are written atomically every save_every optimizer steps. A kill
between saves loses work after the last checkpoint. No claim of deterministic
CUDA kernels is made; RNG and loader ordering are restored for controlled resume.

COCO must be downloaded and verified before submitting text training; see the
[text chapter](../docs/24_text_conditioning.md). `TEXT_BASE_MODEL` and
`IMAGE_DATA_ROOT` make local paths portable. LoRA uses a pretrained denoiser;
`--config-name text_compact` trains our denoiser from scratch with frozen external
codec/text components. The [curriculum](../docs/00_start_here.md) notebooks are CPU
lessons and do not require cluster submission.
