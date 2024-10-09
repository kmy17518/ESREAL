<div align="center">
  <img src="./assets/logo.png" width="20%">
</div>

# 👻 ESREAL

[[`Paper`](https://arxiv.org/abs/2403.16167)] [[`Code`](https://github.com/kmy17518/ESREAL)] [[`Citation`](#black_nib-Citation)]

Official PyTorch implementation of ESREAL. For details, see the paper **[ESREAL: Exploiting Semantic Reconstruction to Mitigate Hallucinations in Vision-Language Models](https://arxiv.org/abs/2403.16167)**.

Authors: Minchan Kim, Minyeong Kim, Junik Bae, Suhwan Choi, Sungkyung Kim, Buru Chang

## 🔥 News
- **`2024/10/10`**: Code is available now.

## 🛠️ Installation

```sh
git clone https://github.com/kmy17518/ESREAL.git
git submodule update --init --recursive
poetry install
```

## 🤖 Reward Server

```sh
sh scripts/reward_server.sh
```

## 🚆 Training

```sh
sh scripts/train.sh
```

## 📝 Inference

```sh
sh scripts/infer.sh
```

## 🙌 Contribution Guide

### Branch Naming

- Feature Branches: For new features.
- Bugfix Branches: For fixing bugs.
- Hotfix Branches: For urgent fixes to production.
- Release Branches: For preparing a new production release.
- Chore Branches: For maintenance tasks.

### Commit Message Convention

For more detail, refer to https://www.conventionalcommits.org/en/v1.0.0/

- Types: Describes the category of the change.
    - feat: A new feature for a specific project.
    - fix: A bug fix for a specific project.
    - docs: Documentation changes.
    - style: Code style changes (formatting, missing semi-colons, etc.).
    - refactor: Code refactoring without changing functionality.
    - perf: Performance improvements.
    - test: Adding or updating tests.
    - chore: Changes to the build process or auxiliary tools and libraries.

## ❣️ Acknowledgement

We are very grateful for the great previous works including LLaVA, InstructBLIP, mPLUG-Owl2, SDXL, SDXL Turbo, HyperSDXL, and Grounding DINO.

## ✒️ Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.  

```bibtex
@article{esreal,
      title={Exploiting Semantic Reconstruction to Mitigate Hallucinations in Vision-Language Models}, 
      author={Minchan Kim and Minyeong Kim and Junik Bae and Suhwan Choi and Sungkyung Kim and Buru Chang},
      year={2024},
      url={https://arxiv.org/abs/2403.16167}, 
}
```
