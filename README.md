
<div align="center">

<!-- 1. LOGO & TITLE -->

<img src="static/images/logo.png" width="120px">

# BiManiBench

### A Hierarchical Benchmark for Evaluating Bimanual Coordination of Multimodal Large Language Models

[**Xin Wu**](https://openreview.net/profile?id=%7EXin_Wu13)`<sup>`1*`</sup>` · [**Zhixuan Liang**](https://liang-zx.github.io/)`<sup>`2*`</sup>` · [**Yue Ma**](https://mayuelala.github.io/)`<sup>`3,4†`</sup>` · [**Mengkang Hu**](https://aaron617.github.io/)`<sup>`2`</sup>` · [**Zhiyuan Qin**](https://openreview.net/profile?id=~Zhiyuan_Qin1)`<sup>`4`</sup>` · [**Xiu Li**](https://openreview.net/profile?id=~Xiu_Li1)`<sup>`1†`</sup>`

`<sup>`1`</sup>`Tsinghua University &nbsp;&nbsp; `<sup>`2`</sup>`The University of Hong Kong &nbsp;&nbsp; `<sup>`3`</sup>`HKUST &nbsp;&nbsp; `<sup>`4`</sup>`Beijing Innovation Center of Humanoid Robotics
`<br>`
`<sup>`*`</sup>`Equal Contribution &nbsp;&nbsp; `<sup>`†`</sup>`Corresponding Authors

---

<!-- 2. QUICK LINKS -->

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=googlechrome&logoColor=white)](https://bimanibench.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://bimanibench.github.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📢 News

- **[2026-02]** BiManiBench project page and preprint are released!
- **[2026-02]** Evaluation code and benchmark assets are coming soon.

---

## 💡 About BiManiBench

**BiManiBench** is the **first hierarchical benchmark** specifically designed to systematically evaluate the **bimanual coordination** capabilities of Multimodal Large Language Models (MLLMs).

While current research in embodied AI has made significant strides in single-arm manipulation, bimanual coordination remains a formidable challenge. It requires more than just parallel execution; it demands **rigorous spatiotemporal synchronization** and **dynamic role assignment** to navigate complex kinematic constraints and prevent self-collisions.

### 🌟 Key Features

- **Hierarchical Evaluation Framework:** Deconstructs bimanual tasks into three levels:
  - **Tier 1 (Dual-Arm Spatial Reasoning):** Fundamental workspace awareness and arm allocation.
  - **Tier 2 (High-Level Action Planning):** Long-horizon reasoning under diverse coordination modes (parallel & sequential).
  - **Tier 3 (Low-Level End-Effector Control):** Direct generation of fine-grained, 16-DoF continuous poses.
- **Vision-Driven Agent Pipeline:** A structured closed-loop reasoning framework where the MLLM functions as a central "brain" for iterative perception, reasoning, and action.
- **Extensive Empirical Study:** Analysis of over **30+ state-of-the-art models**, revealing a significant **"reasoning-actuation gap"** in current foundation models.

<p align="center">
  <img src="static/images/framework_overview.png" width="95%">
  <br>
  <em>Figure 1: The hierarchical evaluation framework of BiManiBench.</em>
</p>

<p align="center">
  <img src="static/images/agent_pipeline.png" width="95%">
  <br>
  <em>Figure 2: The vision-driven agent pipeline for multimodal perception and reasoning.</em>
</p>

---

## 🛠️ Installation & Quick Start

*Code release is in progress. Stay tuned!*

```bash
# Example commands (coming soon)
git clone https://github.com/bimanibench/BiManiBench.git
cd BiManiBench
pip install -r requirements.txt
```


## 🖋️ Citation

**If you find BiManiBench useful in your research, please cite our work:**

**code**

```
@article{wu2026bimanibench,
  author    = {Wu, Xin and Liang, Zhixuan and Ma, Yue and Hu, Mengkang and Qin, Zhiyuan and Li, Xiu},
  title     = {{BiManiBench}: A Hierarchical Benchmark for Evaluating Bimanual Coordination of Multimodal Large Language Models},
  journal   = {arXiv preprint},
  year      = {2026},
}
```
