# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

import torch
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from torchmultimodal.models.flava.model import (
    flava_model_for_classification,
    flava_model_for_pretraining,
)
from transformers.optimization import get_cosine_schedule_with_warmup
from flava.metrics import R1_mAP_eval

def get_optimizers_for_lightning(
    model: torch.nn.Module,
    learning_rate: float,
    adam_eps: float,
    adam_weight_decay: float,
    adam_betas: Tuple[int, int],
    warmup_steps: int,
    max_steps: int,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=adam_betas,
        eps=adam_eps,
        weight_decay=adam_weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class FLAVAPreTrainingLightningModule(LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.0002,
        adam_eps: float = 1.0e-08,
        adam_weight_decay: float = 0.01,
        adam_betas: Tuple[int, int] = (0.9, 0.999),
        warmup_steps: int = 2000,
        max_steps: int = 450000,
        **flava_pretraining_kwargs: Any,
    ):
        super().__init__()
        self.model = flava_model_for_pretraining(**flava_pretraining_kwargs)
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.adam_betas = adam_betas
        self.adam_weight_decay = adam_weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def training_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        losses = output.losses
        total_loss = 0
        for key in losses:
            if losses[key] is not None:
                total_loss += losses[key]
                self.log(f"train/losses/{key}", losses[key], prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        losses = output.losses
        total_loss = 0
        for key in losses:
            if losses[key] is not None:
                total_loss += losses[key]
                self.log(
                    f"validation/losses/{key}", losses[key], prog_bar=True, logger=True
                )

        return total_loss

    def _step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        if "image" in batch and ("text" in batch or "text_masked" in batch):
            required_embedding = "mm"
        elif "image" in batch:
            required_embedding = "image"
        elif "text" in batch or "text_masked" in batch:
            required_embedding = "text"
        else:
            raise RuntimeError("Batch needs to have either or both 'image' and 'text'.")

        output = self.model(
            image=batch.get("image", None),
            image_for_codebook=batch.get("image_for_codebook", None),
            image_patches_mask=batch.get("image_patches_mask", None),
            text=batch.get("text", None),
            text_masked=batch.get("text_masked", None),
            mlm_labels=batch.get("mlm_labels", None),
            itm_labels=batch.get("itm_labels", None),
            required_embedding=required_embedding,
        )
        return output

    def configure_optimizers(self):
        return get_optimizers_for_lightning(
            self.model,
            self.learning_rate,
            self.adam_eps,
            self.adam_weight_decay,
            self.adam_betas,
            self.warmup_steps,
            self.max_steps,
        )


class FLAVAClassificationLightningModule(LightningModule):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 0.0002,
        adam_eps: float = 1.0e-08,
        adam_weight_decay: float = 0.01,
        adam_betas: Tuple[int, int] = (0.9, 0.999),
        warmup_steps: int = 2000,
        max_steps: int = 450000,
        **flava_classification_kwargs: Any,
    ):
        super().__init__()
        self.model = flava_model_for_classification(
            num_classes, **flava_classification_kwargs
        )
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.adam_weight_decay = adam_weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.adam_betas = adam_betas
        self.metrics = Accuracy()
        num_query = 3368 #real_market,len(query) = 3368
        self.evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm="yes")

    def training_step(self, batch, batch_idx):
        output, accuracy = self._step(batch, batch_idx)
        self.log("train/losses/classification", output.loss, prog_bar=True, logger=True)
        self.log(
            "train/accuracy/classification",
            accuracy,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return output.loss



    def validation_step(self, batch, batch_idx):
        # feat, accuracy = self._step_val(batch, batch_idx)
        # self.log(
        #     "validation/losses/classification", output.loss, prog_bar=True, logger=True
        # )
        # self.log(
        #     "validation/accuracy/classification",
        #     accuracy,
        #     prog_bar=True,
        #     logger=True,
        #     sync_dist=True,
        # )

        # return output.loss
        feat = self._step_val(batch, batch_idx)
        return feat




    def validation_epoch_end(self, validation_step_outputs):
        cmc, mAP, _, _, _, _, _ = self.evaluator.compute()
        # self.log(
        #     "Validation Results", validation_step_outputs, prog_bar=True, logger=True
        # )
        # type(validation_step_outputs):list, `list` values cannot be logged


        # logger.info("Validation Results - Epoch: {}".format(epoch))
        # logger.info("mAP: {:.1%}".format(mAP))
        self.log(
            "mAP", mAP, prog_bar=True, logger=True
        )
        self.log("Rank-1", cmc[1 - 1], prog_bar=True, logger=True)
        self.log("Rank-5", cmc[5 - 1], prog_bar=True, logger=True)
        self.log("Rank-10", cmc[10 - 1], prog_bar=True, logger=True)
        # for r in [1, 5, 10]:
        #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        print("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


    def _step(self, batch, batch_idx):
        if "image" in batch and ("text" in batch or "text_masked" in batch):
            required_embedding = "mm"
        elif "image" in batch:
            required_embedding = "image"
        elif "text" in batch or "text_masked" in batch:
            required_embedding = "text"
        else:
            raise RuntimeError("Batch needs to have either or both 'image' and 'text'.")
        # import pdb; pdb.set_trace()
        labels = batch["image_ids"]
        output = self.model(
            image=batch.get("image", None),
            text=batch.get("text", None),
            required_embedding=required_embedding,
            labels=labels,
        )

        # if isinstance(output.logits, list):
        #     acc = (output.logits[0].max(1)[1] == labels).float().mean()
        # else:
        #     acc = (output.logits.max(1)[1] == labels).float().mean()

        accuracy = self.metrics(output.logits.max(1)[1], labels)

        return output, accuracy

    def _step_val(self, batch, batch_idx):
        if batch_idx == 0:
            self.evaluator.reset()


        if "image" in batch and ("text" in batch or "text_masked" in batch):
            required_embedding = "mm"
        elif "image" in batch:
            required_embedding = "image"
        elif "text" in batch or "text_masked" in batch:
            required_embedding = "text"
        else:
            raise RuntimeError("Batch needs to have either or both 'image' and 'text'.")
        # import pdb; pdb.set_trace()
        labels = batch["image_ids"]
        camids = batch["cam_ids"]
        feat = self.model(
            image=batch.get("image", None),
            text=batch.get("text", None),
            required_embedding=required_embedding,
            labels=labels,
        )

        self.evaluator.update((feat, labels, camids))
        # accuracy = self.metrics(logits.max(1)[1], labels)

        return feat

    def configure_optimizers(self):
        return get_optimizers_for_lightning(
            self.model,
            self.learning_rate,
            self.adam_eps,
            self.adam_weight_decay,
            self.adam_betas,
            self.warmup_steps,
            self.max_steps,
        )
