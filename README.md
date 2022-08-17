Img2Mol: inferring molecules from pictures
==========================================
![Img2Mol](Img2Mol.png)
Welcome to Img2Mol! :wave:.

## Overview
 Here we provide the implementation of the `img2mol` model using [PyTorch](https://github.com/pytorch/pytorch) and [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) for training and inference, along with an exemplary jupyter notebook.
 
This repository is organized as follows:
* `examples/`: contains example images to apply our proposed model on
* `img2mol/`: contains necessary python modules for our proposed model
* `model/`: stores the trained model weights as pickled files. The download-link will be provided in future soon

## Installation

#### Environment
Create a new environment:
```bash
git clone git@github.com:bayer-science-for-a-better-life/Img2Mol.git
cd Img2Mol
conda env create -f environment.yml
conda activate img2mol
pip install .
```
## Download Model Weights
* [Download and unzip the CDDD model](https://drive.google.com/u/0/uc?id=1oyknOulq_j0w9kzOKKIHdTLo5HphT99h&export=download)
* Move the directory *default_model* to `path/to/anaconda3/envs/img2mol/lib/python3.6/site-packages/cddd/data/`


* [Download the img2mol model here](https://drive.google.com/file/d/1pk21r4Zzb9ZJkszJwP9SObTlfTaRMMtF/view)

* Move the downloaded file *model.ckpt* into the `path/to/anaconda3/envs/img2mol/lib/python3.6/site-packages/cddd/model/` directory.  

Alternatively, we provide a bash script that will download and move the file automatically (it still needs to be copied to the package directory).
```bash
bash download_model.sh
```
If you have problems downloading the file using the bash script, please manually download the file using the browser.

## Examples
### Usage
**The Img2MolInference object is instanciated with three parameters:**
* model_ckpt (str ): Model path (defaults to None) - If it is not specified, the model file is assumed to be placed in `path/to/anaconda3/envs/img2mol/lib/python3.6/site-packages/cddd/model/`
* device (str): Device used for inference (defaults to `"cuda:0" if torch.cuda.is_available() else "cpu"`,
* local_cddd (bool): Indicates whether or not to use the local cddd installation (Defaults to `False`)

**An instanciated Img2MolInference object can be called with the path of an image to determine the SMILES representation of the depicted molecule**
```
from img2mol.inference import *

img2mol = Img2MolInference(local_cddd=True)
result = img2mol(filepath="examples/digital_example1.png")
```

**Check the example notebook `example_inference.ipynb` to see how the inference class can be used.**

## Reference
Please cite our manuscript if you use our model in your work.

D.-A. Clevert, T. Le, R. Winter, F. Montanari, Chem. Sci., 2021, [DOI: 10.1039/D1SC01839F](https://doi.org/10.1039/D1SC01839F)

## Img2Mol Code License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

##  Model Parameters License
The Img2Mol parameters are made available for non-commercial use only, under the terms of the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. You can find details at: https://creativecommons.org/licenses/by-nc/4.0/legalcode
