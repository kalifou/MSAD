########################################################################
#
# @title : Finetuning Script
# @description: Finetune pretrained MSAD models on ESA-ADB with:
#               - LoRA
#               - Layer-wise discriminative learning rates
#               - Cosine annealing with warmup
#               - Stochastic Weight Averaging (SWA)
#               - Gradient clipping
#               - Mixed precision training
#
# Example:
#   python3 finetune_deep_model.py \
#       --path=data/TSB_512/ \
#       --model=convnet \
#       --weights=results/weights/supervised/convnet_default_512/model_30012023_173428 \
#       --params=models/configuration/convnet_default.json
# ########################################################################

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, top_k_accuracy_score, confusion_matrix
)
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
import re
import json
import copy
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


try:
    from utils.timeseries_dataset import create_splits, TimeseriesDataset
    from utils.train_deep_model_utils import json_file
    from utils.config import deep_models
    MSAD_AVAILABLE = True
except ImportError:
    print("ERROR: MSAD utils not found. Please ensure 'utils' folder is in PYTHONPATH.")
    MSAD_AVAILABLE = False


class LoRALayer(nn.Module):

    def __init__(self, in_features, out_features, rank=8, alpha=16.0, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x, original_weight, bias=None):
        # Original forward pass
        result = F.linear(x, original_weight, bias)

        # LoRA adaptation: x @ A^T @ B^T
        lora_result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T

        return result + lora_result * self.scaling


class LoRALinear(nn.Module):
    """Wrapper for Linear layer with LoRA"""

    def __init__(self, linear_layer, rank=8, alpha=16.0):
        super().__init__()
        self.linear = linear_layer

        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )

    def forward(self, x):
        return self.lora(x, self.linear.weight, self.linear.bias)


def apply_lora_to_model(model, rank=8, alpha=16.0, verbose=True):
    """
    Apply LoRA to all Linear layers except the final classifier.
    """
    total_before = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)

    to_replace = []

    # classifier layer names:
    # sit.py -> cls_layer
    # resnet.py -> final
    # inception_time.py -> linear
    # convnet.py -> fc1
    classifier_keywords = ['classifier', 'cls_layer',
                           'final', 'fc1', 'head', 'output']

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this linear layer belongs to a classifier block
            is_classifier = False
            for k in classifier_keywords:
                if k in name:
                    is_classifier = True
                    break

            if is_classifier:
                continue

            to_replace.append((name, module))

    # Apply replacements
    lora_applied = []
    for name, module in to_replace:
        # Get parent module
        *parent_path, child_name = name.split('.')
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)

        # Replace
        lora_linear = LoRALinear(module, rank=rank, alpha=alpha)
        setattr(parent, child_name, lora_linear)
        lora_applied.append(name)

    total_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"\n{'='*70}")
        print(f"LoRA Applied to {len(lora_applied)} layers")
        print(f"{'='*70}")
        print(f"Trainable parameters: {total_before:,} -> {total_after:,}")
        if total_before > 0:
            print(f"Reduction: {100*(1 - total_after/total_before):.1f}%")
        print(f"{'='*70}\n")

    return model, lora_applied


def get_layer_wise_optimizer(model, config):
    """
    Create optimizer with discriminative learning rates based on relative depth.

    Logic based on provided model files:
    1. Head (High LR): Identified by names like 'cls_layer' (SIT), 'final' (ResNet), 'fc1' (ConvNet).
    2. Backbone (Variable LR): Split into Early (first 40%) and Middle (next 60%).
    """
    head_params = []
    backbone_params_list = []  # List of (name, param)

    head_keywords = ['cls_layer', 'final', 'fc1', 'linear']

    # 1. Separate Head and Backbone
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_head = False
        # Check top-level module name mainly
        top_level = name.split('.')[0]
        if any(k == top_level for k in head_keywords):
            is_head = True
        elif 'classifier' in name:  # Fallback
            is_head = True

        if is_head:
            head_params.append(
                {'params': [param], 'lr': config['head_lr'], 'name': f"head/{name}"})
        else:
            backbone_params_list.append((name, param))

    # 2. Split Backbone into Early and Middle based on parameter index
    n_backbone = len(backbone_params_list)
    # First 40% is "early" (embedding/early convs)
    split_idx = int(n_backbone * 0.4)

    early_params = []
    middle_params = []

    for i, (name, param) in enumerate(backbone_params_list):
        if i < split_idx:
            early_params.append(
                {'params': [param], 'lr': config['backbone_lr'], 'name': f"early/{name}"})
        else:
            middle_params.append(
                {'params': [param], 'lr': config['middle_lr'], 'name': f"middle/{name}"})

    optimizer_params = head_params + middle_params + early_params

    print(f"\nLayer-wise Learning Rates:")
    print(f"{'='*70}")
    print(
        f"Early Layers  (LR {config['backbone_lr']:.2e}): {len(early_params)} params")
    print(
        f"Middle Layers (LR {config['middle_lr']:.2e}): {len(middle_params)} params")
    print(
        f"Head Layers   (LR {config['head_lr']:.2e}): {len(head_params)} params")
    print(f"{'='*70}\n")

    optimizer = optim.AdamW(
        optimizer_params,
        betas=config['betas'],
        eps=config['eps'],
        weight_decay=config['weight_decay']
    )

    return optimizer


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-7):
    """Cosine annealing with linear warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine annealing
        progress = (current_step - num_warmup_steps) / \
            max(1, num_training_steps - num_warmup_steps)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))

        # Scale to [min_lr, 1.0]
        return max(min_lr, cosine_decay)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EarlyStopping:

    def __init__(self, patience=7, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


def train_epoch(model, dataloader, criterion, optimizer, scheduler, config, epoch):
    """Train for one epoch with modern techniques"""
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    scaler = GradScaler(
    ) if config['use_amp'] and torch.cuda.is_available() else None

    pbar = tqdm(
        dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")

    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(config['device']).float()
        labels = labels.to(config['device']).long()

        if scaler is not None:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            # Gradient clipping
            if config['gradient_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config['gradient_clip'])

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            if config['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config['gradient_clip'])

            optimizer.step()
            optimizer.zero_grad()

        # Update scheduler if step-based
        if scheduler is not None and config['scheduler_step'] == 'batch':
            scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % config['log_interval'] == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{total_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%",
                'lr': f"{current_lr:.2e}"
            })

    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': 100.0 * correct / total
    }

    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, config, desc="Validation"):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, labels in tqdm(dataloader, desc=desc, leave=False):
        inputs = inputs.to(config['device']).float()
        labels = labels.to(config['device']).long()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        probs = F.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = 100.0 * accuracy_score(all_labels, all_preds)

    # Top-k accuracies
    try:
        n_classes = all_probs.shape[1]
        top3_acc = 100.0 * \
            top_k_accuracy_score(all_labels, all_probs, k=min(
                3, n_classes), labels=np.arange(n_classes))
        top5_acc = 100.0 * \
            top_k_accuracy_score(all_labels, all_probs, k=min(
                5, n_classes), labels=np.arange(n_classes))
    except Exception as e:
        top3_acc = top5_acc = accuracy

    f1_macro = f1_score(all_labels, all_preds,
                        average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds,
                           average='weighted', zero_division=0)

    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

    return metrics, all_preds, all_labels, all_probs


def finetune_deep_model(
    data_path,
    model_name,
    pretrained_path,
    model_parameters_file,
    output_dir='results/finetuned',
    # Data args
    split_per=0.7,
    seed=42,
    read_from_file=None,
    batch_size=32,
    # Finetuning args
    use_lora=True,
    lora_rank=16,
    lora_alpha=32.0,
    use_layer_wise_lr=True,
    backbone_lr=1e-5,
    middle_lr=5e-5,
    head_lr=1e-3,
    # Training args
    num_epochs=30,
    warmup_epochs=2,
    weight_decay=0.01,
    gradient_clip=1.0,
    # SWA args
    use_swa=True,
    swa_start=20,
    swa_lr=5e-6,
    # Other args
    early_stopping_patience=7,
    eval_model=True
):

    print("\n" + "="*80)
    print("MSAD MODEL FINETUNING - Modern Techniques")
    print("="*80)

    if not MSAD_AVAILABLE:
        raise ImportError(
            "MSAD utilities not found. Cannot proceed without data loading logic.")

    # Setup

    window_size = int(re.search(r'\d+', str(data_path)).group())
    print(f"\nWindow size: {window_size}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/weights", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{model_name}_{window_size}_{timestamp}"

    print(f"\n{'='*80}")
    print("DATA LOADING")
    print(f"{'='*80}")

    train_set, val_set, test_set = create_splits(
        data_path,
        split_per=split_per,
        seed=seed,
        read_from_file=read_from_file,
    )

    training_data = TimeseriesDataset(data_path, fnames=train_set)
    val_data = TimeseriesDataset(data_path, fnames=val_set)
    test_data = TimeseriesDataset(data_path, fnames=test_set)

    print(f"Train samples: {len(training_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    class_weights = training_data.get_weights_subset(device)
    num_classes = len(class_weights)

    train_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"\n{'='*80}")
    print("MODEL SETUP")
    print(f"{'='*80}")

    model_parameters = json_file(model_parameters_file)

    # Adjust for window size - for SignalTransformer (sit.py)
    if 'original_length' in model_parameters:
        model_parameters['original_length'] = window_size
    if 'timeseries_size' in model_parameters:
        model_parameters['timeseries_size'] = window_size

    # Adjust number of classes
    if 'num_classes' in model_parameters:
        model_parameters['num_classes'] = num_classes

    try:
        model = deep_models[model_name.lower()](**model_parameters)
    except KeyError:
        raise ValueError(
            f"Model '{model_name}' not found in deep_models registry.")

    model = model.to(device)

    # Load pretrained weights
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"\nLoading pretrained weights from: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)

            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=False)
            print("✓ Pretrained weights loaded successfully")

        except Exception as e:
            print(f"⚠ Could not load pretrained weights: {e}")
            print("  Training from scratch...")
    else:
        print(f"\nNo pretrained weights found at: {pretrained_path}")
        print("Training from scratch...")

    # Apply LoRA

    if use_lora:
        print(f"\n{'='*80}")
        print("APPLYING LoRA (Low-Rank Adaptation)")
        print(f"{'='*80}")
        print(f"Rank: {lora_rank}, Alpha: {lora_alpha}")

        model, lora_layers = apply_lora_to_model(
            model,
            rank=lora_rank,
            alpha=lora_alpha,
            verbose=True
        )

    # Setup Training

    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*80}")

    # Configuration dictionary
    config = {
        'device': device,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'backbone_lr': backbone_lr,
        'middle_lr': middle_lr,
        'head_lr': head_lr,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': weight_decay,
        'gradient_clip': gradient_clip,
        'use_amp': torch.cuda.is_available(),
        'log_interval': 10,
        'scheduler_step': 'epoch'  # or 'batch'
    }

    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(
        f"Learning rates - Backbone: {backbone_lr:.2e}, Middle: {middle_lr:.2e}, Head: {head_lr:.2e}")
    print(f"Weight decay: {weight_decay}")
    print(f"Gradient clip: {gradient_clip}")
    print(f"Mixed precision: {config['use_amp']}")

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Optimizer with layer-wise learning rates
    if use_layer_wise_lr:
        optimizer = get_layer_wise_optimizer(model, config)
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=head_lr,
            betas=config['betas'],
            eps=config['eps'],
            weight_decay=weight_decay
        )
        print(f"\nUsing single LR: {head_lr:.2e}")

    # Learning rate scheduler
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr=1e-7
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=0.001,
        mode='max'
    )

    # Stochastic Weight Averaging
    swa_model = None
    swa_scheduler = None
    if use_swa and swa_start < num_epochs:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        print(f"\nSWA enabled: starts at epoch {swa_start}, LR={swa_lr:.2e}")

    # Training Loop

    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}\n")

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_top3_acc': [],
        'val_top5_acc': [],
        'val_f1_macro': [],
        'learning_rates': []
    }

    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            scheduler if config['scheduler_step'] == 'batch' else None,
            config, epoch
        )

        # Validate
        val_metrics, _, _, _ = evaluate(
            model, val_loader, criterion, config, desc=f"Epoch {epoch+1} [Val]"
        )

        # Update scheduler (epoch-based)
        if scheduler is not None and config['scheduler_step'] == 'epoch':
            scheduler.step()

        # SWA update
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_top3_acc'].append(val_metrics['top3_accuracy'])
        history['val_top5_acc'].append(val_metrics['top5_accuracy'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(
            f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, "
              f"Top-3: {val_metrics['top3_accuracy']:.2f}%, Top-5: {val_metrics['top5_accuracy']:.2f}%")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = copy.deepcopy(model.state_dict())

            checkpoint_path = f"{output_dir}/weights/{run_name}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'val_top3_accuracy': val_metrics['top3_accuracy'],
                'val_top5_accuracy': val_metrics['top5_accuracy'],
                'config': config,
            }, checkpoint_path)

            print(f"  ✓ New best model saved (Val Acc: {best_val_acc:.2f}%)")

        if early_stopping(val_metrics['accuracy'], epoch):
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            print(
                f"  Best validation accuracy: {early_stopping.best_score:.2f}% at epoch {early_stopping.best_epoch+1}")
            break

    # Final Evaluation

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Update BN statistics with SWA if used
    if use_swa and swa_model is not None:
        print("\nUpdating SWA model batch norm statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        model = swa_model

    print(f"\n{'='*80}")
    print("FINAL EVALUATION")
    print(f"{'='*80}\n")

    # Test set evaluation
    if eval_model:
        test_metrics, test_preds, test_labels, test_probs = evaluate(
            model, test_loader, criterion, config, desc="Test Set"
        )

        print(f"Test Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"  Top-3 Accuracy: {test_metrics['top3_accuracy']:.2f}%")
        print(f"  Top-5 Accuracy: {test_metrics['top5_accuracy']:.2f}%")
        print(f"  F1-Macro: {test_metrics['f1_macro']:.4f}")
        print(f"  F1-Weighted: {test_metrics['f1_weighted']:.4f}")

        # Save test metrics
        test_results = {
            'model': model_name,
            'window_size': window_size,
            'timestamp': timestamp,
            **test_metrics
        }

        results_df = pd.DataFrame([test_results])
        results_df.to_csv(
            f"{output_dir}/logs/{run_name}_test_results.csv", index=False)

        # Confusion matrix
        cm = confusion_matrix(test_labels, test_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        plt.title('Test Set Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(
            f"{output_dir}/plots/{run_name}_confusion_matrix.png", dpi=150)
        plt.close()

    # Save Training History

    history_df = pd.DataFrame(history)
    history_df.to_csv(f"{output_dir}/logs/{run_name}_history.csv", index=False)

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Val Top-1')
    axes[0, 1].plot(history['val_top3_acc'], label='Val Top-3', linestyle='--')
    axes[0, 1].plot(history['val_top5_acc'], label='Val Top-5', linestyle=':')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Learning rate
    axes[1, 0].plot(history['learning_rates'])
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)

    # Empty 4th plot (CM is saved separately now)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/{run_name}_training_curves.png", dpi=150)
    plt.close()

    print(f"\n{'='*80}")
    print(f"✓ Finetuning complete!")
    print(f"  Run name: {run_name}")
    print(f"  Best model: {output_dir}/weights/{run_name}_best.pth")
    print(f"  Logs: {output_dir}/logs/")
    print(f"  Plots: {output_dir}/plots/")
    print(f"{'='*80}\n")

    return model, history


# Command Line Interface


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='finetune_deep_model',
        description='Finetune MSAD models with modern techniques (LoRA, layer-wise LR, etc.)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to dataset (must contain window size as number)')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model name (e.g., cnn, resnet, transformer)')
    parser.add_argument('-w', '--weights', type=str, required=True,
                        help='Path to pretrained weights (.pth file)')
    parser.add_argument('-pa', '--params', type=str, required=True,
                        help='JSON file with model parameters')

    # Data arguments
    parser.add_argument('-s', '--split', type=float, default=0.7,
                        help='Train split percentage')
    parser.add_argument('-se', '--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('-f', '--file', type=str, default=None,
                        help='File with predefined splits')
    # REMOVED: --dummy-labels (No longer supported)

    # Finetuning arguments
    parser.add_argument('--use-lora', action='store_true', default=False,
                        help='Apply LoRA for parameter-efficient finetuning')
    parser.add_argument('--lora-rank', type=int, default=16,
                        help='LoRA rank (higher = more expressive)')
    parser.add_argument('--lora-alpha', type=float, default=32.0,
                        help='LoRA alpha scaling factor')

    parser.add_argument('--layer-wise-lr', action='store_true', default=False,
                        help='Use discriminative learning rates')
    parser.add_argument('--backbone-lr', type=float, default=1e-5,
                        help='Learning rate for early layers')
    parser.add_argument('--middle-lr', type=float, default=5e-5,
                        help='Learning rate for middle layers')
    parser.add_argument('--head-lr', type=float, default=1e-3,
                        help='Learning rate for classifier head')

    # Training arguments
    parser.add_argument('-b', '--batch', type=int, default=32,
                        help='Batch size')
    parser.add_argument('-ep', '--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='Warmup epochs')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Gradient clipping value')

    # SWA
    parser.add_argument('--use-swa', action='store_true', default=False,
                        help='Use Stochastic Weight Averaging')
    parser.add_argument('--swa-start', type=int, default=20,
                        help='Epoch to start SWA')
    parser.add_argument('--swa-lr', type=float, default=5e-6,
                        help='SWA learning rate')

    # Other
    parser.add_argument('--early-stopping', type=int, default=7,
                        help='Early stopping patience')
    parser.add_argument('-o', '--output', type=str, default='results/finetuned',
                        help='Output directory')
    parser.add_argument('-e', '--eval-true', action='store_true',
                        help='Evaluate on test set')

    args = parser.parse_args()

    # Run finetuning
    finetune_deep_model(
        data_path=args.path,
        model_name=args.model,
        pretrained_path=args.weights,
        model_parameters_file=args.params,
        output_dir=args.output,
        # Data
        split_per=args.split,
        seed=args.seed,
        read_from_file=args.file,
        batch_size=args.batch,
        # LoRA
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        # Layer-wise LR
        use_layer_wise_lr=args.layer_wise_lr,
        backbone_lr=args.backbone_lr,
        middle_lr=args.middle_lr,
        head_lr=args.head_lr,
        # Training
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        # SWA
        use_swa=args.use_swa,
        swa_start=args.swa_start,
        swa_lr=args.swa_lr,
        # Other
        early_stopping_patience=args.early_stopping,
        eval_model=args.eval_true
    )
