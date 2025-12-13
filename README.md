# KGS-UNet
KGS-UNet is the innovative model proposed in this paper. All other comparative models, the unified training framework, and the datasets (DRIVE, CHASEDB1, ISIC18) are sourced from U-Bench (https://github.com/FengheTan9/U-Bench). We would like to express our special gratitude to the U-Bench team for providing the unified training framework.
#Abstract
Abstract—Accurate segmentation of small anatomical struc-
tures under limited supervision remains challenging for U-shaped
networks, whose skip connections typically fuse encoder features
without selection. We propose KGS-UNet, which repurposes
Kolmogorov–Arnold Networks (KANs) as active gates at feature-
transfer interfaces. Specifically,the KAN-Gated Skip (KGS) mod-
ule performs pixel-wise, noise-aware fusion between encoder and
decoder features, complementing this,the AttnDown block couples
strided convolution with KAN-driven channel–spatial attention
to preserve thin structures during downsampling. An optional
bottleneck aggregates multi-scale and graph-enhanced context
with minimal overhead. Using a single configuration, KGS-UNet
attains top-ranked performance on DRIVE [3] and CHASEDB1
[4] and maintains stable rankings on ISIC 2018 [5] within the
U-Bench protocol, while consistently outperforming KAN-based
baselines. These results suggest that spline-based nonlinearity
at cross-scale connections is a viable design for small-structure
medical segmentation.
