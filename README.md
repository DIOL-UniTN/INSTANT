# INSTANT: Inference-Aware Fast Feedforward Networks

This repository contains two main components:

## torch

- Contains algorithmic experimentations using PyTorch.
- Utilizes Hydra for configuration management.
- Integrates MLflow for experiment tracking and workflow management.

## mcu

- Contains code for deployment on edge devices using microcontrollers.
- Focuses on enabling efficient inference and integration with hardware.

Please read the README files inside the `torch` and `mcu` directories to reproduce experiments.

# Citation

If you find this work useful, please consider citing:

```bibtex
@article{10.1145/3815117,
author = {Kilic, Renan Beran and Yildirim, Kasim Sinan and Iacca, Giovanni},
title = {INSTANT: Inference-Aware Fast Feedforward Networks},
year = {2026},
issue_date = {July 2026},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {25},
number = {4},
issn = {1539-9087},
url = {https://doi.org/10.1145/3815117},
doi = {10.1145/3815117},
abstract = {Many embedded applications have strict energy, memory, and time constraints, making neural network (NN) inference particularly challenging. Recently, a novel NN architecture, called Fast Feedforward Networks (FFFs), has been proposed to achieve inference with extremely lightweight computational demands and minimal latency. Yet, compared to feedforward networks with similar sizes, FFFs still lag behind in terms of performance, indicating that they do not utilize all of their parameters effectively. In this article, we explore a possible reason for this performance gap: the uncertainty in how samples are assigned to the network’s leaves. We attempt to overcome this challenge by making FFFs’ training inference-aware, hence introducing Inference-Aware Fast Feedforward Networks (IAFFFs). We imitate FFFs’ inference during training by using a step activation function alongside the traditional sigmoid activation function. We test different aware scheduling methods, which we dub “awareness scheduler”, to adjust the balance between the two activation functions during training, and examine how different schedules impact the model’s performance. Additionally, we employ leaf-weight virtualization with inference-aware retraining to compress our models so they can fit onto edge devices. We further employ an iterative compression approach to find an optimal awareness scheduler for compression to minimize performance drop due to compression. We experiment with different model sizes on various microcontrollers (MCUs) with different memory constraints to observe the latency and energy consumption introduced by the compression algorithm.},
journal = {ACM Trans. Embed. Comput. Syst.},
month = jun,
articleno = {62},
numpages = {38},
keywords = {Fast feedforward networks, inference-aware models, efficient machine learning}
}
```
