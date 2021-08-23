# `img2mol` module structure
This directory consists of the necessary python scripts to perform inference tasks with the `img2mol` model.
The list below summarizes each module:


* `cddd_server.py`
    * class for utilizing th CDDD encoder-decoder described by [Winter et al. (2019)](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc04175j#!divAbstract)
    * note that the implemented model class is licensed under the CC BY-NC 4.0 license and only applicable in non-commercial setting
* `model.py`
    * Model implementation of the `img2mol` as described in our paper. We use Pytorch Lightning for model training, but essentially, only using PyTorch is also possible
* `inference.py`
    * inference class that can be used for predicting the SMILES representation based on an image representation. By default, the model weights are randomly initialized and when instantatiating the inference class, a model checkpoint can be used for loading trained weights.
    * The provided model weights are licensed under CC BY-NC 4.0 and only applicable for non-commercial usage
