# PatchCore backbone weights

The weights file is not in git (`*.pth` is ignored). Put it here:

    patchcore/weights/wide_resnet50_2_imagenet.pth

It is the torchvision ImageNet WideResNet50 (`IMAGENET1K_V1`). To get it:

    curl -L -o patchcore/weights/wide_resnet50_2_imagenet.pth https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth

The `95faca4d` in the file name is the start of its sha256, so `sha256sum` of the file must begin with it.

`weights_path` in `config_patchcore.yaml` points here (`/jidoka/code/jt.train.jade/patchcore/weights/...`).

The weights were trained on ImageNet. Check the ImageNet and torchvision license terms before commercial use.
