import logging
import math
import os
import re
from typing import Any, Callable, Optional
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm, trange

from transformers import Trainer
from transformers import PreTrainedModel
# from transformers import is_wandb_available
from transformers import TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transformers.trainer_utils import (
    EvalPrediction,
)

from transformers.trainer_callback import (
    TrainerCallback,
)

logger = logging.getLogger(__name__)

class IETrainer(Trainer):
    """
    IETrainer is inheritated from from transformers.Trainer, optimized for IE tasks.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        use_crf: Optional[bool]=False
    ):
        super(IETrainer, self).__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks
        )
        self.use_crf = use_crf

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method (or :obj:`create_optimizer`
        and/or :obj:`create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        self.create_scheduler(num_training_steps)

    def create_scheduler(self, num_training_steps: int):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            warmup_steps = (
                self.args.warmup_steps
                if self.args.warmup_steps > 0
                else math.ceil(num_training_steps * self.args.warmup_ratio)
            )

            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:

            no_decay = ["bias", "LayerNorm.weight"]
            if self.use_crf:

                crf = "crf"
                crf_lr = self.args.crf_learning_rate
                logger.info(f"Learning rate for CRF: {crf_lr}")
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in self.model.named_parameters()
                            if (not any(nd in n for nd in no_decay)) and (crf not in n)
                        ],
                        "weight_decay": self.args.weight_decay
                    },
                    {
                        "params": [p for p in self.model.crf.parameters()],
                        "weight_decay": self.args.weight_decay,
                        "lr": crf_lr
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters()
                            if any(nd in n for nd in no_decay) and (not crf not in n)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in self.model.named_parameters()
                            if not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters()
                            if any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]


            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    def get_optimizers(
        self,
        num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.
        """
        if self.optimizers is not None:
            return self.optimizers

        no_decay = ["bias", "LayerNorm.weight"]
        if self.use_crf:
            crf = "crf"
            crf_lr = self.args.crf_learning_rate
            logger.info(f"Learning rate for CRF: {crf_lr}")
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if (not any(nd in n for nd in no_decay)) and (crf not in n)
                    ],
                    "weight_decay": self.args.weight_decay
                },
                {
                    "params": [p for p in self.model.crf.parameters()],
                    "weight_decay": self.args.weight_decay,
                    "lr": crf_lr
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay) and (not crf not in n)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps
        )

        return optimizer, scheduler

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self._prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output['metrics'])

        return output

    def predict(self, test_dataset: Dataset) -> Dict:
        test_dataloader = self.get_test_dataloader(test_dataset)

        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(
        self,
        dataloader: DataLoader,
        description: str
    ) -> Dict:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`
        Works both with or without labels.
        """
        model = self.model
        batch_size = dataloader.batch_size

        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)

        model.eval()

        eval_losses: List[float] = []
        preds_ids = []
        label_ids = []

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(
                inputs.get(k) is not None
                for k in ["labels", "lm_labels", "masked_lm_labels"]
            )

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            mask = inputs["decoder_mask"].to(torch.bool)
            preds = model.decode(logits, mask=mask)
            preds_ids.extend(preds)
            if inputs.get("labels") is not None:
                labels = [inputs["labels"][i, mask[i]].tolist() \
                            for i in range(inputs["labels"].shape[0])]
                label_ids.extend(labels)
                assert len(preds) == len(labels)
                assert len(preds[0]) == len(labels[0])

        if self.compute_metrics is not None and \
                len(preds_ids) > 0 and \
                len(label_ids) > 0:
            metrics = self.compute_metrics(preds_ids, label_ids)
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics['eval_loss'] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return {'predictions': preds_ids, 'label_ids': label_ids, 'metrics': metrics}


    def log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        
        if self.state.global_step is None:
            # when logging evaluation metrics without training
            self.state.global_step = 0

        output = {**logs, **{"step": self.state.global_step}}

        if iterator is not None:
            iterator.write(output)
        
        else:
            logger.info(
                {k:round(v, 4) if isinstance(v, float) else v for k, v in output.items()}
            )

        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

