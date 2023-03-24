Img2Mol: inferring molecules from pictures
==========================================
![Img2Mol](Img2Mol.png)
Welcome to Img2Mol! :wave:.

:point_right: For the Img2Mol web app switch to the "deployment-example" branch.

## Overview
 Here we provide the implementation of the `img2mol` model using [PyTorch](https://github.com/pytorch/pytorch) and [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) for training and inference, along with an exemplary jupyter notebook.
 
This repository is organized as follows:
* `examples/`: contains example images to apply our proposed model on
* `img2mol/`: contains necessary python modules for our proposed model
* `model/`: stores the trained model weights as pickled files. The download-link will be provided in future soon

## Installation
#### Requirements
```
python=3.8.5
pip=20.2.4
notebook=6.4.2
pillow=8.0.1
numpy=1.19.2
rdkit=2020.03.1
cudatoolkit=11.0
torchvision=0.8.0
torchaudio=0.7.0
pytorch=1.7.0
pytorch-lightning=1.0.8
```

#### Environment
Create a new environment:
```bash
git clone git@github.com:bayer-science-for-a-better-life/Img2Mol.git
cd Img2Mol
conda env create -f environment.yml
conda activate img2mol
pip install .
```
*If you want to run Img2Mol as a standalone version with a locally loaded CDDD model instead of sending requests to our CDDD server, install the environment from `environment.local-cddd.yml` instead of `environment.yml`*
## Download Model Weights
You can download the trained parameters for the default model (~2.4GB) as described in our paper using the following link:
<a href="https://drive.google.com/file/d/1pk21r4Zzb9ZJkszJwP9SObTlfTaRMMtF/view" target="_blank">https://drive.google.com/file/d/1pk21r4Zzb9ZJkszJwP9SObTlfTaRMMtF/view </a>.  
Please move the downloaded file `model.ckpt` into the `model/` directory.  

If you are working with the local CDDD installation, please * [download and unzip the CDDD model](https://drive.google.com/u/0/uc?id=1oyknOulq_j0w9kzOKKIHdTLo5HphT99h&export=download) and ove the directory *default_model* to `path/to/anaconda3/envs/img2mol/lib/python3.6/site-packages/cddd/data/`

Alternatively, we provide a bash script that will download and move the file automatically.
```bash
bash download_model.sh
```
If you have problems downloading the file using the bash script, please manually download the file using the browser.

## Examples
Check the example notebook `example_inference.ipynb` to see how the inference class can be used. A demonstration of the usage with the usage with the local CDDD model is demonstrated in `example_inference_local_cddd.ipynb`.

## Reference
Please cite our manuscript if you use our model in your work.

D.-A. Clevert, T. Le, R. Winter, F. Montanari, Chem. Sci., 2021, [DOI: 10.1039/D1SC01839F](https://doi.org/10.1039/D1SC01839F)

## Img2Mol Code License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

##  Model Parameters License
The Img2Mol parameters are made available for non-commercial use only, under the terms of the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. You can find details at: https://creativecommons.org/licenses/by-nc/4.0/legalcode
