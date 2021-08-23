# Copyright 2021 Machine Learning Research @ Bayer AG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torchvision import transforms

from rdkit import Chem

from typing import Optional
import random
import numpy as np

from PIL import Image, ImageOps, ImageEnhance

from img2mol.model import Img2MolPlModel
from img2mol.cddd_server import CDDDRequest

"""
Inference Class for Img2Mol Model.
By default, the class instantiation will not use any model checkpoint.
The Img2Mol model parameters are made available for non-commercial use only, under the terms of the
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.
You can find details at: https://creativecommons.org/licenses/by-nc/4.0/legalcode
"""


class Img2MolInference(object):
    """
    Inference Class
    """
    def __init__(self, model_ckpt: Optional[str] = None, device: str = "cpu"):
        super(Img2MolInference, self).__init__()
        self.device = device
        print("Initializing Img2Mol Model with random weights.")
        self.model = Img2MolPlModel()
        if model_ckpt is not None:
            print(f"Loading checkpoint: {model_ckpt}")
            self.model = self.model.load_from_checkpoint(model_ckpt)

        print("Setting to `self.eval()`-mode.")
        self.model.eval()
        print(f"Sending model to `{self.device}` device.")
        self.model.to(self.device)
        print("Succesfully created Img2Mol Inference class.")

    """
    Class methods for image preprocessing
    """
    @classmethod
    def read_imagefile(cls, filepath: str) -> Image.Image:
        img = Image.open(filepath, "r")

        if img.mode == "RGBA":
            bg = Image.new('RGB', img.size, (255, 255, 255))
            # Paste image to background image
            bg.paste(img, (0, 0), img)
            return bg.convert('L')
        else:
            return img.convert('L')

    @classmethod
    def fit_image(cls, img: Image):
        old_size = img.size
        desired_size = 224
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = img.resize(new_size, Image.BICUBIC)
        new_img = Image.new("L", (desired_size, desired_size), "white")
        new_img.paste(img, ((desired_size - new_size[0]) // 2,
                            (desired_size - new_size[1]) // 2))

        new_img = ImageOps.expand(new_img, int(np.random.randint(5, 25, size=1)), "white")
        return new_img

    @classmethod
    def transform_image(cls, image: Image):
        image = cls.fit_image(image)
        img_PIL = transforms.RandomRotation((-15, 15), resample=3, expand=True, center=None, fill=255)(image)
        img_PIL = transforms.ColorJitter(brightness=[0.75, 2.0], contrast=0, saturation=0, hue=0)(img_PIL)
        shear_value = np.random.uniform(0.1, 7.0)
        shear = random.choice([[0, 0, -shear_value, shear_value], [-shear_value, shear_value, 0, 0],
                               [-shear_value, shear_value, -shear_value, shear_value]])
        img_PIL = transforms.RandomAffine(0, translate=None, scale=None,
                                          shear=shear, resample=3, fillcolor=255)(img_PIL)
        img_PIL = ImageEnhance.Contrast(ImageOps.autocontrast(img_PIL)).enhance(2.0)
        img_PIL = transforms.Resize((224, 224), interpolation=3)(img_PIL)
        img_PIL = ImageOps.autocontrast(img_PIL)
        img_PIL = transforms.ToTensor()(img_PIL)
        return img_PIL

    def read_image_to_tensor(self, filepath: str,
                             repeats: int = 50):
        extension = filepath.split(".")[-1] in ("jpg", "jpeg", "png")
        if not extension:
            return "Image must be jpg or png format!"
        image = self.read_imagefile(filepath)
        images = torch.cat([torch.unsqueeze(self.transform_image(image), 0) for _ in range(repeats)], dim=0)
        images = images.to(self.device)

        return images

    def __call__(self, filepath: str,
                 cddd_server: CDDDRequest,
                 return_cddd: bool = False,
                 ) -> dict:
        images = self.read_image_to_tensor(filepath, repeats=50)
        with torch.no_grad():
            cddd = self.model(images).detach().cpu().numpy()

        # take the median cddd prediction out of `repeats` predictions
        cddd = np.median(cddd, axis=0)

        smiles = cddd_server.cddd_to_smiles(cddd.tolist())
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        # if the molecule is valid, i.e. can be parsed with the rdkit
        if mol:
            can_smiles = Chem.MolToSmiles(mol)
            can_mol = Chem.MolFromSmiles(can_smiles)
        else:
            print("Image translation failed.")
            can_smiles = None
            can_mol = None

        if not return_cddd:
            cddd = None

        return {"filepath": filepath,
                "cddd": cddd, "smiles": can_smiles, "mol": can_mol
                }


    def predict(self, filepath: str,
                cddd_server: CDDDRequest,
                return_cddd: bool = False) -> dict:
        return self.__call__(filepath, cddd_server, return_cddd)



if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    img2mol = Img2MolInference(model_ckpt=None,
                               device=device)

    cddd_server = CDDDRequest(host="http://ec2-18-157-240-87.eu-central-1.compute.amazonaws.com")

    example = "examples/example1.png"

    res = img2mol(filepath=example, cddd_server=cddd_server)
