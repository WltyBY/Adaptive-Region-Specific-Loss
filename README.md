# Adaptive-Region-Specific-Loss
Thanks for the work by: Chen Y, Yu L, Wang J Y, et al. Adaptive Region-Specific Loss for Improved Medical Image Segmentation. https://ieeexplore.ieee.org/abstract/document/10163830

The code in adaptive_region_adaptivepool.py use AdaptiveAvgPool to do the task. This allows users to set any number of boxes per axis. However, it's better not to set the number of boxes per axis larger than the two inputs themselves!!

What's more, if you want to set the size of per region as you want, the adaptive_region_pool.py will meet your requirement.
