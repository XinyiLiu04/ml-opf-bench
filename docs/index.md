## Benchmarking Machine Learning Paradigms for Optimal Power Flow
Xinyi Liu, Xuan He, Yize Chen

SCUT, HKUST, UAlberta 

In this work, we propose ML-OPF-Bench, which is to the authors' knowledge the pioneering work on benchmarking performance of ML-based OPF solvers. ML-OPF-Bench provides:

## Introduction

The Optimal Power Flow (OPF) problem is one of the fundamental optimization tasks in power system operations and planning, which is frequently solved on a regular basis. As modern power systems grow in complexity—with increasing grid variability, and real-time operational demands—solving OPF efficiently and reliably under uncertainty has become more challenging than ever. In response, the research community has increasingly turned to machine learning (ML) approaches for OPF, motivated by the promise of fast inference, model-free approximation, and robust performance in scenarios where traditional optimization solvers may struggle.

Despite rapid progress, the ML-for-OPF landscape is fragmented and difficult to compare in a rigorous, reproducible manner. Existing studies often use different test cases, data generation pipelines, input/output parameterizations, and evaluation metrics—making it hard to assess progress or determine which approaches are best suited for real-world deployment. To address this gap, we introduce ML-OPF-Bench, a unified benchmarking suite for evaluating ML models on OPF tasks across multiple dimensions of performance.

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
