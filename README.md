# CelestialNet: Image Segmentation for Orbital Scenes Using Domain-Specific Cues
## Introduction
In this project, we aim to utilize depth estimation maps in conjunction with edge detection to enhance satellite segmentation efforts. EfficientNet-B5 was chosen as the backbone for our model due to its strong feature extraction capabilities at high resolutions, allowing it to effectively capture both fine-grained spatial details for heatmap prediction and global context necessary for accurate depth estimation, all while maintaining computational efficiency. Depth estimation was accomplished through an AdaBins module due to its ability to adaptively discretize depth ranges and produce more accurate, edge-preserving depth maps compared to traditional regression-based approaches. Keypoint estimation was implemented by producing heatmaps the same size as the input images that represented the probability that each pixel was part of a keypoint in the image. These outputs were combined into a single loss function to benefit from multi-task learning. We found that the keypoint maps of satellite images did not capture edges in the way we augmented the training dataset, thereby weakening the segmentation results. However, it was found that multi-task learning and the combined loss function led to better segmentation.
## Repository
Below you can find the descriptions of the code we were working on. Other files and modules are imported from AdaBins implementation -- you can find more on it in the link that is provided in the report. In order to reproduce our results, just see the notebook below and go through the cells.
- `preprocess_depths.ipynb` -- notebook containing data processing and augumentations code.
- `multimodalfusion.ipynb` -- notebook containing combined code for depth and edge models.
- `seg_deeplabv3_mobilenet.ipynb` -- notebook containing code for baseline segmentation model.
- `heatmap_module.ipynb` -- notebook containing code for edge detection model.
- `DepthEstimation.ipynb` -- notebook containing code for depth estimation network.
- `/models_for_joint_learning` -- module containing depth and edge models.
- `/baseline_utils` -- module containing utils for baseline model preprocessing, training and combined loss.
