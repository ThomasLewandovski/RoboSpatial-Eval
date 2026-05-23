# SpatialBot

Official implementation of **SpatialBot**, developed for the **CVPR 2026 ERA Workshop RoboSpatial Challenge**.

SpatialBot is a spatial reasoning system built upon RoboBrain2.5 for evaluating embodied spatial understanding on RoboSpatial-Home.

---

## Team SpatialBot

Contributors:

- Haijiao Zhao
- Kehan Meng
- Jianhua Yang
- Zhizhen Cai
- Dong An
- Yixiang Chen
- Yan Huang
- Liang Wang

---

## Features

- RoboBrain2.5 spatial reasoning backbone
- Automatic HuggingFace checkpoint download
- Automatic local checkpoint cache
- RoboSpatial-Home benchmark support
- Configuration reasoning
- Compatibility reasoning
- Context pointing reasoning
- Automatic coordinate normalization for pointing tasks

---

## Installation

Clone repository:

```bash
git clone https://github.com/ThomasLewandovski/RoboSpatial-Eval.git

cd SpatialBot
```

Create environment:

```bash
conda create -n spatialbot python=3.10

conda activate spatialbot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Model Checkpoints

SpatialBot automatically downloads model checkpoints from Hugging Face.

First run:

```

HF → checkpoints/

```

Later runs:

```

local cache → directly loaded

```

Configure checkpoint source in:

```text
config.yaml
```

Example:

```yaml
robobrain25:

model_repo:

YOUR_HF_REPO

local_model_dir:

./checkpoints/SpatialBot

```

---

## Run Evaluation

Run partial benchmark:

```bash
python main.py robobrain25 --config config.yaml --limit 10
```

Full evaluation:

```bash
python main.py robobrain25 --config config.yaml
```

---

## Supported Benchmark

Dataset:

```

chanhee-luke/RoboSpatial-Home

```

Tasks:

- Configuration
- Compatibility
- Context

---

## Pointing Output Convention

SpatialBot predicts coordinates using RoboBrain2.5 output convention:

```

(732,411)

↓

(0.732,0.411)

```

Coordinates are normalized automatically into:

```

[0,1]

```

before evaluation.

---

## Project Structure

```text

SpatialBot/

├── main.py

├── config.yaml

├── models.py

├── checkpoints/

├── results/

├── requirements.txt

└── README.md

```

---

## Citation

Coming soon.