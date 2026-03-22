# Reference Papers

## Core Architecture References

### Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering
- **Authors:** Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, Bo Dai
- **Venue:** CVPR 2024 (Highlight)
- **ArXiv:** https://arxiv.org/abs/2312.00109
- **Code:** https://github.com/city-super/Scaffold-GS
- **Summary:** Introduces anchor-based representation where anchor points distribute local 3D Gaussians with attributes predicted on-the-fly by small MLPs conditioned on viewing direction and distance. Addresses redundancy in vanilla 3DGS through anchor growing/pruning guided by neural Gaussian importance. Achieves faster convergence, fewer primitives, and superior quality on challenging scenes.

### Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians
- **Authors:** Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, Bo Dai
- **Venue:** IEEE TPAMI 2025
- **ArXiv:** https://arxiv.org/abs/2403.17898
- **Code:** https://github.com/city-super/Octree-GS
- **Summary:** Organizes anchor Gaussians into a hierarchical octree for Level-of-Detail decomposition, selecting detail level based on viewing distance. Uses progressive training with next-level growth and opacity/view-frequency pruning. Yields consistent real-time rendering across varying viewpoints and scales.

## Compression References

### ContextGS: Compact 3D Gaussian Splatting with Anchor Level Context Model
- **Authors:** Yufei Wang, Zhihao Li, Lanqing Guo, Wenhan Yang, Alex C. Kot, Bihan Wen
- **Venue:** NeurIPS 2024
- **ArXiv:** https://arxiv.org/abs/2405.20721
- **Code:** https://github.com/wyf0912/ContextGS
- **Summary:** First autoregressive context model at the anchor level for 3DGS compression. Divides anchors into hierarchical levels where uncoded anchors are predicted from coarser-level coded anchors for entropy modeling. Achieves 100x reduction vs vanilla 3DGS and 15x vs Scaffold-GS while maintaining quality.

### CompGS: Efficient 3D Scene Representation via Compressed Gaussian Splatting
- **Authors:** Xiangrui Liu, Xinju Wu, Pingping Zhang, Shiqi Wang, Zhu Li, Sam Kwong
- **Venue:** ACM Multimedia 2024
- **ArXiv:** https://arxiv.org/abs/2404.09458
- **Code:** https://github.com/LiuXiangrui/CompGS
- **Summary:** Hybrid primitive structure using anchor primitives to predict remaining primitives stored as compact residuals. Incorporates rate-constrained optimization to eliminate redundancy within the hybrid structure, optimizing the bitrate-quality tradeoff.

### PCGS: Progressive Compression of 3D Gaussian Splatting
- **Authors:** Yihang Chen, Mengyao Li, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, Jianfei Cai
- **Venue:** AAAI 2026 (Oral)
- **ArXiv:** https://arxiv.org/abs/2503.08511
- **Code:** https://github.com/YihangChen-ee/PCGS
- **Summary:** Progressive masking strategy that incrementally adds anchors while refining existing ones, paired with progressive quantization that gradually reduces step sizes. Leverages existing quantization results for probability prediction across progressive levels. Enables true progressive decoding for bandwidth-adaptive streaming.
