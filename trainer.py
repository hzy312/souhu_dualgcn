#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：souhu_dualgcn 
@File ：trainer.py
@Author ：Huang ZiYang
@Date ：2022/4/11 16:09 
"""
from argparse import Namespace
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, TRAIN_DATALOADERS, EVAL_DATALOADERS
from transformers import AutoModel
from model import DualGCNBertClassifier
from torchmetrics import F1Score
from transformers import AdamW, get_linear_schedule_with_warmup
from data import SOUHUDataModule


class SOUHUModule(pl.LightningModule):
    def __init__(self, opt: Namespace):
        super(SOUHUModule, self).__init__()
        self.opt = opt
        self.save_hyperparameters()
        self.data_module = SOUHUDataModule(self.opt)
        self.bert = AutoModel.from_pretrained(opt.plm_path)
        self.model = DualGCNBertClassifier(self.bert, self.opt)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.F1 = F1Score(num_classes=opt.polarities_dim)
        self._initialize()

    def _initialize(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    torch.nn.init.xavier_uniform_(p)  # xavier_uniform_
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.opt.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),
                          lr=self.opt.lr)
        t_total = len(self.train_dataloader()) // self.opt.accumulate_grad_batches * self.opt.max_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.opt.warmup_steps,
                                                    num_training_steps=t_total)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _compute_loss_and_predictions(self, batch):
        label= batch['label']
        logits, penal = self.model(batch)
        loss = self.criterion(logits, label)
        loss = loss + penal
        y_pred = logits.argmax(dim=-1)
        return loss, y_pred

    def training_step(self, batch, batch_idx):
        label = batch['label']
        loss, y_pred = self._compute_loss_and_predictions(batch)
        return {'loss': loss, 'y_pred': y_pred, 'y_true': label}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        y_pred = torch.cat([o['y_pred'] for o in outputs])
        y_true = torch.cat([o['y_true'] for o in outputs])
        f1 = self.F1(y_pred, y_true)
        self.log('train_f1', f1, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        label = batch['label']
        loss, y_pred = self._compute_loss_and_predictions(batch)
        return {'loss': loss, 'y_pred': y_pred, 'y_true': label}

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        y_pred = torch.cat([o['y_pred'] for o in outputs])
        y_true = torch.cat([o['y_true'] for o in outputs])
        f1 = self.F1(y_pred, y_true)
        self.log('val_f1', f1, prog_bar=True)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        idx = batch['id']
        entity = batch['entity']
        logits, _ = self.model(batch)
        pred = logits.argmax(dim=-1)
        results = list(zip(idx, entity, pred))
        results = [[int(ins[0]), ins[1], int(ins[2])-2] for ins in results]
        return results


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.data_module.train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.data_module.val_dataloader()
