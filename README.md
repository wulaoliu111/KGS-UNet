# KGS-UNet
KGS-UNet is the innovative model proposed in this paper. All other comparative models, the unified training framework, and the datasets (DRIVE, CHASEDB1, ISIC18) are sourced from U-Bench (https://github.com/FengheTan9/U-Bench). We would like to express our special gratitude to the U-Bench team for providing the unified training framework.
# Abstract
Accurate segmentation of small anatomical struc-
tures remains challenging for U-shaped networks because skip
connections often fuse encoder and decoder features without
explicit selection. We propose KGS-UNet, which repurposes
Kolmogorov–Arnold Networks (KANs) as learnable gates at skip
interfaces. The KAN-Gated Skip (KGS) module performs pixel-
wise routing using spline-parameterized gates with noise-aware
lower bounds to suppress inconsistent encoder activations while
preserving decoding cues. In addition, the AttnDown module
integrates KAN-driven channel attention and spatial gating to
better retain thin structures during downsampling.Benchmarked
against over 70 mainstream U-Net variants under a unified
training framework, KGS-UNet secures the top rank on the
challenging, small-scale DRIVE and CHASEDB1 datasets. Specif-
ically, it surpasses competing mainstream models by margins of
1.5% in both IoU and F1 scores. Furthermore, compared with
existing KAN-based U-Net variants that primarily utilize KANs
as activation replacements, KGS-UNet achieves a substantial
improvement of over 6 percentage points in IoU on DRIVE.
These results suggest that spline-based gating at skip interfaces
is an effective design choice for medical segmentation.
