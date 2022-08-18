from typing import Dict
from typing import List
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
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
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [[128, 7, 3, 4], [256, 5, 1, 1], [384, 5, 1, 1], 'M', [384, 3, 1, 1], [384, 3, 1, 1], 'M', [512, 3, 1, 1],
          [512, 3, 1, 1], [512, 3, 1, 1], 'M'],
    'B': [[256, 8, 2, 2], [256, 5, 1, 0], 'M', [384, 3, 1, 1], [384, 3, 1, 1], 'M', [512, 3, 1, 1], [512, 3, 1, 1]],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class pic2Smiles(pl.LightningModule):

    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate

        self.features = make_layers(cfgs['A'], batch_norm=False)
        # self.maxpool = nn.AdaptiveMaxPool2d((7, 7))
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
        # x = self.maxpool(x)
        x = torch.flatten(x, 1)
        # scores = []
        # for _ in range(20):
        # scores.append(self.classifier(x))
        # scores.append(x)
        # x = torch.mean(torch.stack(scores).float(), dim=0)
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
        # training_step defined the train loop. It is independent of forward
        x, cddd = batch
        cddd_hat = self(x)
        loss = F.mse_loss(cddd_hat, cddd)
        # difference = cddd_hat - cddd
        # loss = torch.mean(torch.log(torch.cosh(difference)))
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, cddd = batch
        cddd_hat = self(x)
        loss = F.mse_loss(cddd_hat, cddd)
        # difference = cddd_hat - cddd
        # loss = torch.mean(torch.log(torch.cosh(difference)))
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, cddd = batch
        cddd_hat = self(x)
        loss = F.mse_loss(cddd_hat, cddd)
        # difference = cddd_hat - cddd
        # loss = torch.mean(torch.log(torch.cosh(difference)))
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


def train_dataloader(data_train, batch_size):
    return DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)


def val_dataloader(data_val, batch_size):
    return DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)


def test_dataloader(data_test, batch_size):
    return DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
