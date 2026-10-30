Code copied from https://github.com/facebookresearch/dinov2 (Apache License 2.0, see LICENSE).
Only the files needed to build the DINOv2 backbone are kept: layers/, models/vision_transformer.py, hub/backbones.py, hub/utils.py.
models/__init__.py is empty on purpose (the original one pulls in the training code).
The weights (.pth) are in ../../weights/.
