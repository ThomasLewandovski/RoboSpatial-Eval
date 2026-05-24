# SpatialBot

Official implementation of **SpatialBot**, developed for the **CVPR 2026 ERA Workshop RoboSpatial Challenge**.

SpatialBot is a spatial reasoning system built upon RoboBrain2.5 for evaluating embodied spatial understanding on RoboSpatial-Home.

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

## Features

- RoboBrain2.5 spatial reasoning backbone
- Automatic ModelScope checkpoint download
- Automatic local checkpoint cache
- RoboSpatial-Home benchmark support
- Configuration reasoning task support
- Compatibility reasoning task support
- Context pointing reasoning task support
- Automatic coordinate normalization for pointing tasks

## Installation

Clone repository:

```bash
git clone https://github.com/ThomasLewandovski/RoboSpatial-Eval.git
cd RoboSpatial-Eval
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

## Model Checkpoints

SpatialBot automatically downloads model checkpoints from ModelScope and caches them locally.

Model weights: [Lewandovski/SpatialBot](https://modelscope.cn/models/Lewandovski/SpatialBot)

Configure checkpoint source in:

[`config.yaml`](config.yaml)

Example:

```yaml
robobrain25:
  model_repo: "modelscope://Lewandovski/SpatialBot"
  local_model_dir: "./checkpoints/robobrain25"
```

Download behavior:

```text
First run:  ModelScope -> checkpoints/
Later runs: local cache -> directly loaded
```

## Run Evaluation

Full evaluation:

```bash
python main.py robobrain25 --config config.yaml
```

## Supported Benchmark

Dataset:

```text
chanhee-luke/RoboSpatial-Home
```

Tasks:

- Configuration
- Compatibility
- Context

## Pointing Output Convention

SpatialBot predicts coordinates using RoboBrain2.5 output convention:

```text
(732,411)
|
v
(0.732,0.411)
```

Coordinates are normalized automatically into:

```text
[0,1]
```

before evaluation.

## Project Structure

```text
RoboSpatial-Eval/
├── main.py
├── config.yaml
├── models.py
├── checkpoints/
├── results/
├── requirements.txt
└── README.md
```

## Citation

Coming soon.
