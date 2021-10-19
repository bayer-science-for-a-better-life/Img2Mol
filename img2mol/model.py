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
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Union, List, Optional


MODEL_CONFIGS: List = [[128, 7, 3, 4],
                       [256, 5, 1, 1],
                       [384, 5, 1, 1],
                       'M',
                       [384, 3, 1, 1],
                       [384, 3, 1, 1],
                       'M',
                       [512, 3, 1, 1],
                       [512, 3, 1, 1],
                       [512, 3, 1, 1],
                       'M']


def make_layers(cfg: Optional[List[Union[str, int]]] = None,
                batch_norm: bool = False) -> nn.Sequential:
    """
    Helper function to create the convolutional layers for the Img2Mol model to be passed into a nn.Sequential module.
    :param cfg: list populated with either a str or a list, where the str object refers to the pooling method and the
                list object will be unrolled to obtain the convolutional-filter parameters.
                Defaults to the `MODEL_CONFIGS` list.
    :param batch_norm: boolean of batch normalization should be used in-between conv2d and relu activation.
                       Defaults to False
    :return: torch.nn.Sequential module as feature-extractor
    """
    if cfg is None:
        cfg = MODEL_CONFIGS

    layers: List[nn.Module] = []

    in_channels = 1
    for v in cfg:
        if v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                units, kern_size, stride, padding = v
                conv2d = nn.Conv2d(in_channels, units, kernel_size=kern_size, stride=stride, padding=padding)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(units), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = units

    model = nn.Sequential(*layers)
    return model


class Img2MolPlModel(pl.LightningModule):
    """
    Wraps the Img2Mol model into pytorch lightning for easy training and inference
    """
    def __init__(self, learning_rate: float = 1e-4, batch_norm: bool = False):
        super().__init__()
        self.learning_rate = learning_rate

        # convolutional NN for feature extraction
        self.features = make_layers(cfg=MODEL_CONFIGS, batch_norm=batch_norm)
        # fully-connected network for classification based on CNN feature extractor
        self.classifier = nn.Sequential(
            nn.Linear(512 * 9 * 9, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 512),
            nn.Tanh(),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def training_step(self, batch, batch_idx):
        x, cddd = batch
        cddd_hat = self(x)
        loss = F.mse_loss(cddd_hat, cddd)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, cddd = batch
        cddd_hat = self(x)
        loss = F.mse_loss(cddd_hat, cddd)
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, cddd = batch
        cddd_hat = self(x)
        loss = F.mse_loss(cddd_hat, cddd)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    pl_model = Img2MolPlModel()
    print(pl_model)
