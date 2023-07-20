# Adaptive-Region-Specific-Loss
Thanks for the work by: Chen Y, Yu L, Wang J Y, et al. Adaptive Region-Specific Loss for Improved Medical Image Segmentation.
The code in adaptive_region_adaptivepool.py use AdaptiveAvgPool to do the task. This allows users to use any number of boxes. However, it's better not to set the number of boxes per axis larger than the two inputs themselves!!
