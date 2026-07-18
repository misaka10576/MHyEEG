from tqdm import tqdm
from earlystopping import EarlyStopping
from contextlib import nullcontext
from collections.abc import Mapping
import torch
import time
import torch.nn as nn
import torch.optim.lr_scheduler as sched
import wandb
from sklearn.metrics import f1_score   

class Trainer():
    def __init__(self, net, optimizer, epochs,
                      use_cuda=True, gpu_num=0,
                      checkpoint_folder="./checkpoints",
                      max_lr=0.1, min_mom=0.7,
                      max_mom=0.99, l1_reg=False,
                      num_classes=3,
                      sample_weights=None,
                      es_mode='max',
                      patience=10,
                      amp=False,
                      amp_dtype='bfloat16',
                      fp32_finetune_epochs=0,
                      allow_tf32=False,
                      wandb_log_interval=20):

        self.optimizer = optimizer
        self.epochs = epochs
        self.use_cuda = use_cuda
        self.gpu_num = gpu_num
        self.checkpoints_folder = checkpoint_folder
        # self.max_lr = max_lr
        self.min_mom = min_mom,
        self.max_mom = max_mom
        self.l1_reg = l1_reg
        self.num_classes = num_classes
        self.es_mode = es_mode
        self.patience = patience
        self.use_amp = bool(amp and use_cuda)
        self.amp_dtype_name = amp_dtype
        self.fp32_finetune_epochs = fp32_finetune_epochs
        self.allow_tf32 = bool(allow_tf32 and use_cuda)
        self.wandb_log_interval = max(1, wandb_log_interval)
        self._fast_precision_enabled = None

        if self.fp32_finetune_epochs < 0 or self.fp32_finetune_epochs > self.epochs:
            raise ValueError("fp32_finetune_epochs must be between 0 and epochs.")
        if self.amp_dtype_name not in ('bfloat16', 'float16'):
            raise ValueError("amp_dtype must be 'bfloat16' or 'float16'.")

        self.amp_dtype = (
            torch.bfloat16
            if self.amp_dtype_name == 'bfloat16'
            else torch.float16
        )
        if (
            self.use_amp
            and self.amp_dtype == torch.bfloat16
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "The selected GPU does not support bfloat16 AMP. "
                "Use amp_dtype=float16 or disable AMP."
            )
        self.scaler = torch.amp.GradScaler(
            'cuda',
            enabled=self.use_amp and self.amp_dtype == torch.float16,
        )

        sample_weights = torch.tensor(sample_weights, dtype=torch.float32) if len(sample_weights)>0 else None
        self.criterion = nn.CrossEntropyLoss(weight=sample_weights)
        self.val_criterion = nn.CrossEntropyLoss()
        
        if self.use_cuda:
            if sample_weights is not None:
                self.criterion.weight = sample_weights.clone().detach().cuda('cuda:%i' %self.gpu_num)

            print(f"Running on GPU?", self.use_cuda, "- gpu_num: ", self.gpu_num)
            self.net = net.cuda('cuda:%i' %self.gpu_num)
            
        else:
            self.net = net

        fast_epochs = self.epochs - self.fp32_finetune_epochs
        if self.use_amp:
            print(
                f"Precision schedule: epochs 1-{fast_epochs} use "
                f"{self.amp_dtype_name} AMP"
            )
            if self.fp32_finetune_epochs > 0:
                print(
                    f"Precision schedule: epochs {fast_epochs + 1}-{self.epochs} "
                    "use full FP32"
                )
        elif self.allow_tf32:
            print(f"Precision schedule: epochs 1-{fast_epochs} allow TF32")
        else:
            print("Precision schedule: full FP32 for all epochs")

    def _autocast_context(self, enabled):
        if not enabled:
            return nullcontext()
        return torch.autocast(
            device_type='cuda',
            dtype=self.amp_dtype,
        )

    def _move_to_device(self, value, device):
        """Move nested multimodal batches to GPU without changing structure."""
        if isinstance(value, torch.Tensor):
            return value.cuda(device, non_blocking=True)
        if isinstance(value, Mapping):
            return {
                key: self._move_to_device(item, device)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item, device) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item, device) for item in value]
        raise TypeError(f"Unsupported batch value type: {type(value).__name__}")

    def _forward_inputs(self, inputs):
        """Support legacy modality lists and named modality dictionaries."""
        if isinstance(inputs, Mapping):
            return self.net(**inputs)
        if isinstance(inputs, (tuple, list)):
            return self.net(*inputs)
        return self.net(inputs)

    def _configure_precision_stage(self, fast_precision_enabled):
        if self._fast_precision_enabled == fast_precision_enabled:
            return

        if not self.use_cuda:
            self._fast_precision_enabled = fast_precision_enabled
            print("Using CPU full precision")
            return

        tf32_enabled = self.allow_tf32 and fast_precision_enabled
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
        torch.set_float32_matmul_precision(
            'high' if tf32_enabled else 'highest'
        )
        self._fast_precision_enabled = fast_precision_enabled

        if fast_precision_enabled:
            modes = []
            if self.use_amp:
                modes.append(f"{self.amp_dtype_name} AMP")
            if tf32_enabled:
                modes.append("TF32")
            if modes:
                print(f"Using accelerated precision: {', '.join(modes)}")
            else:
                print("Using full FP32 precision")
        elif self.use_amp or self.allow_tf32:
            print("Switched to full FP32 precision for final fine-tuning")
        else:
            print("Using full FP32 precision")

    def train(self, train_loader, eval_loader, **sched_kwargs):
        
        # name for checkpoint
        run_name = wandb.run.name

        # initialize the early_stopping object
        checkpoint_path = self.checkpoints_folder + "/best_" + run_name + ".pt"
        early_stopping = EarlyStopping(
            patience=self.patience,
            path=checkpoint_path,
            mode=self.es_mode,
        )

        scheduler = sched.OneCycleLR(self.optimizer, epochs=self.epochs, steps_per_epoch=len(train_loader), 
                                         anneal_strategy='linear', cycle_momentum=True, base_momentum=self.min_mom, max_momentum=self.max_mom, 
                                         three_phase=True, **sched_kwargs)
        # scheduler = sched.StepLR(self.optimizer, step_size=5, gamma=0.1)
        
        best_f1 = 0
        best_loss = 0
        best_acc = 0
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            fast_precision_enabled = (
                epoch < self.epochs - self.fp32_finetune_epochs
            )
            amp_enabled = self.use_amp and fast_precision_enabled
            self._configure_precision_stage(fast_precision_enabled)

            start = time.time()
            running_loss_train = 0.0
            running_loss_eval = 0.0
            train_total = 0.0
            train_correct = 0.0
            train_y_pred = torch.empty(0)
            train_y_true = torch.empty(0)
            total = 0.0
            correct = 0.0
            y_pred = torch.empty(0)
            y_true = torch.empty(0)
            
            self.net.train()  # switch net to training setting 
           
            for batch_index, (inputs, labels) in enumerate(
                tqdm(
                    train_loader,
                    total=len(train_loader),
                    desc='Train round',
                    unit='batch',
                    leave=False,
                )
            ):  # for each batch
                if self.use_cuda:
                    device = 'cuda:%i' % self.gpu_num
                    inputs = self._move_to_device(inputs, device)
                    labels = labels.cuda(device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True)

                with self._autocast_context(amp_enabled):
                    outputs = self._forward_inputs(inputs)
                    loss = self.criterion(outputs, labels)

                    if self.l1_reg:
                        print("Adding L1 regularization to A")
                        # Add L1 regularization to A
                        regularization_loss = 0.0
                        for child in self.net.children():
                            for layer in child.modules():
                                if isinstance(layer, PHConv):
                                    for param in layer.a:
                                        regularization_loss += torch.sum(abs(param))
                        loss += 0.001 * regularization_loss

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                scheduler.step()
                if (
                    batch_index % self.wandb_log_interval == 0
                    or batch_index == len(train_loader) - 1
                ):
                    wandb.log({"lr_step": scheduler.get_last_lr()[0]})

                running_loss_train += loss.item()  # save current loss to compute later a mean

                _, predicted = torch.max(outputs.data, 1)  # compute max logits, along dim 1, return (max, id_max==label)
                
                train_total += labels.size(0)  # how much samples seen so far
                train_correct += (predicted == labels).sum().item()  # how much corrected seen so far

                train_y_pred = torch.cat((train_y_pred, predicted.view(predicted.shape[0]).cpu()))
                train_y_true = torch.cat((train_y_true, labels.view(labels.shape[0]).cpu()))
                
            end = time.time()

            train_acc = 100*train_correct/train_total
            train_f1 = f1_score(train_y_true, train_y_pred, average='macro')
            
           
            self.net.eval()  # switch net to evaluate setting 

                
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                 for inputs, labels in tqdm(eval_loader, total=len(eval_loader), desc='Val round', unit='batch', leave=False):   # for each batch
                    if self.use_cuda:
                        device = 'cuda:%i' % self.gpu_num
                        inputs = self._move_to_device(inputs, device)
                        labels = labels.cuda(device, non_blocking=True)

                    with self._autocast_context(amp_enabled):
                        eval_outputs = self._forward_inputs(inputs)
                        eval_loss = self.val_criterion(eval_outputs, labels)
                    running_loss_eval += eval_loss.item()  # save current loss to compute later a mean

                    _, predicted = torch.max(eval_outputs.data, 1)  # compute max logits, along dim 1, return (max, id_max==label)
                    
                    total += labels.size(0)  # how much samples seen so far
                    correct += (predicted == labels).sum().item()  # how much corrected seen so far

                    y_pred = torch.cat((y_pred, predicted.view(predicted.shape[0]).cpu()))
                    y_true = torch.cat((y_true, labels.view(labels.shape[0]).cpu()))

            acc = 100*correct/total
            f1 = f1_score(y_true, y_pred, average='macro')

            # Log metrics
            wandb.log({"train loss": running_loss_train/len(train_loader), "train acc": train_acc, "train f1": train_f1,
                           "val loss": running_loss_eval/len(eval_loader), "val acc": acc, "val f1": f1, "lr": scheduler.get_last_lr()[0],
                           "epoch": epoch+1, "amp enabled": int(amp_enabled),
                           "tf32 enabled": int(self.allow_tf32 and fast_precision_enabled)})

            print('Epoch {:03d}: Loss {:.4f}, Accuracy {:.4f}, F1 score {:.4f} || Val Loss {:.4f}, Val Accuracy {:.4f}, Val F1 score {:.4f}  [Time: {:.4f}]'
                  .format(epoch + 1, running_loss_train/len(train_loader), train_acc, train_f1, running_loss_eval/len(eval_loader), acc, f1, end-start))
            
            if f1 > best_f1:
                best_f1 = f1
                best_loss = running_loss_eval/len(eval_loader)
                best_acc = acc

            # Early stopping
            if self.es_mode == 'max':
                early_stopping(f1, self.net)
            else:
                early_stopping(running_loss_eval/len(eval_loader), self.net)
            if early_stopping.early_stop:
                print(f"Early stopping")
                break
            
        print(f'Finished Training')

        wandb.log({"Best val loss": best_loss})
        wandb.log({"Best val acc": best_acc})
        wandb.log({"Best val f1": best_f1})
        return {
            "checkpoint_path": checkpoint_path,
            "best_val_loss": best_loss,
            "best_val_acc": best_acc,
            "best_val_f1": best_f1,
        }

    def evaluate(self, data_loader, checkpoint_path=None, split="test"):
        """Evaluate named or legacy multimodal inputs, optionally using a checkpoint."""
        if checkpoint_path is not None:
            if self.use_cuda:
                map_location = "cuda:%i" % self.gpu_num
            else:
                map_location = "cpu"
            state_dict = torch.load(
                checkpoint_path,
                map_location=map_location,
                weights_only=True,
            )
            self.net.load_state_dict(state_dict)

        self.net.eval()
        running_loss = 0.0
        total = 0
        correct = 0
        predictions = []
        targets = []
        with torch.no_grad():
            for inputs, labels in tqdm(
                data_loader,
                total=len(data_loader),
                desc=f"{split.capitalize()} round",
                unit="batch",
                leave=False,
            ):
                if self.use_cuda:
                    device = "cuda:%i" % self.gpu_num
                    inputs = self._move_to_device(inputs, device)
                    labels = labels.cuda(device, non_blocking=True)

                outputs = self._forward_inputs(inputs)
                loss = self.val_criterion(outputs, labels)
                running_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predictions.append(predicted.cpu())
                targets.append(labels.cpu())

        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        loss = running_loss / len(data_loader)
        accuracy = 100 * correct / total
        f1 = f1_score(targets, predictions, average="macro")
        print(
            f"{split.capitalize()}: Loss {loss:.4f}, Accuracy {accuracy:.4f}, "
            f"F1 score {f1:.4f}"
        )
        wandb.log(
            {
                f"{split} loss": loss,
                f"{split} acc": accuracy,
                f"{split} f1": f1,
            }
        )
        return {"loss": loss, "accuracy": accuracy, "f1": f1}
