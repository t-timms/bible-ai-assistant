/home/ttimm/miniforge3/envs/bible-orpo/lib/python3.11/site-packages/unsloth/__init__.py:1531: UserWarning: WARNING: Unsloth should be imported before [trl, transformers, peft] to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.

Please restructure your imports with 'import unsloth' at the top of your file.
  from ._gpu_init import *
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
# bible-orpo env provenance snapshot

- snapshot date: 2026-09-10
- python: 3.11.16  (Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39)
- torch: 2.11.0+cu128
- torch.version.cuda: 12.8
- torch.cuda.get_arch_list(): ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
- transformers: 5.5.0
- trl: 0.24.0
- peft: 0.20.0
- unsloth: 2026.8.22
- accelerate: 1.14.0
- xformers: 0.0.35
- torchao: 0.18.0
- bitsandbytes: 0.50.2
- datasets: 4.3.0

## Rebuild

This conda env is `python=3.11` plus a pip install; `requirements.bible-orpo.txt` is the real spec.
The torch/torchvision pins carry a `+cu128` local version, so they need the pytorch cu128 index:

```
conda create -n bible-orpo python=3.11 -y
conda run -n bible-orpo pip install -r docs/env/requirements.bible-orpo.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

See `docs/ORPO_TWO_ENV_SETUP.md` for the two-env rationale and adapter-key caveats.
Risk on rebuild: `unsloth==2026.8.22` / `unsloth_zoo==2026.8.16` are date-pinned and may be yanked;
if so, pin the nearest surviving release and re-verify `torch.cuda.get_arch_list()` includes `sm_120`.
