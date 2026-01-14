########################################################################
# Universal Finetuning Script for All MSAD Architectures
# Supports: ConvNet, ResNet, InceptionTime, Transformer (SIT)
########################################################################

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, confusion_matrix
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
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from utils.timeseries_dataset import create_splits, TimeseriesDataset
    from utils.train_deep_model_utils import json_file
    from utils.config import deep_models
    MSAD_AVAILABLE = True
except ImportError:
    print("ERROR: MSAD utils not found")
    MSAD_AVAILABLE = False


# ============================================================================
# LoRA Implementation
# ============================================================================

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16.0, dropout=0.1):
        super().__init__()
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x, original_weight, bias=None):
        result = F.linear(x, original_weight, bias)
        lora_result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + lora_result * self.scaling


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=8, alpha=16.0):
        super().__init__()
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        self.lora = LoRALayer(linear_layer.in_features,
                              linear_layer.out_features, rank, alpha)

    def forward(self, x):
        return self.lora(x, self.linear.weight, self.linear.bias)


def apply_lora_to_model(model, rank=8, alpha=16.0, verbose=True):
    """Apply LoRA to Linear layers (for Transformers mainly)"""
    total_before = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    to_replace = []
    classifier_keywords = ['classifier', 'cls_layer',
                           'final', 'fc', 'head', 'output', 'linear']

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip final classifier unless in feed-forward block
            is_classifier = any(k in name.lower() for k in classifier_keywords)
            if is_classifier and not any(ff in name.lower() for ff in ['feed_forward', 'mlp', 'ff']):
                continue
            to_replace.append((name, module))

    lora_applied = []
    for name, module in to_replace:
        current_device = next(module.parameters()).device

        *parent_path, child_name = name.split('.')
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)

        lora_layer = LoRALinear(module, rank, alpha).to(current_device)
        setattr(parent, child_name, lora_layer)
        lora_applied.append(name)

    total_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"\n{'='*70}")
        print(f"LoRA Applied to {len(lora_applied)} layers")
        print(f"{'='*70}")
        for layer in lora_applied[:5]:
            print(f"  - {layer}")
        if len(lora_applied) > 5:
            print(f"  ... and {len(lora_applied) - 5} more")
        print(f"\nTrainable: {total_before:,} -> {total_after:,}")
        if total_before > 0:
            print(f"Reduction: {100*(1 - total_after/total_before):.1f}%")
        print(f"{'='*70}\n")

    return model, lora_applied


# ============================================================================
# Architecture Detection
# ============================================================================

def detect_architecture(model):
    """Detect model architecture"""
    name = model.__class__.__name__.lower()
    if 'convnet' in name or 'cnn' in name:
        return 'convnet'
    elif 'resnet' in name:
        return 'resnet'
    elif 'inception' in name:
        return 'inception'
    elif 'transformer' in name or 'sit' in name or 'signal' in name:
        return 'transformer'

    # Infer from structure
    for n, m in model.named_modules():
        if 'transformer' in n.lower() or 'attention' in n.lower():
            return 'transformer'
        if 'resnet' in n.lower() or 'residual' in n.lower():
            return 'resnet'
    return 'convnet'  # Default


# ============================================================================
# Layer Freezing
# ============================================================================

def freeze_layers(model, architecture, freeze_ratio=0.3, verbose=True):
    """Architecture-aware layer freezing"""
    frozen, trainable = 0, 0

    if architecture == 'convnet':
        layers_per_block = 3
        num_blocks = len(model.layers) // layers_per_block
        freeze_blocks = max(1, int(num_blocks * freeze_ratio))
        freeze_up_to = freeze_blocks * layers_per_block

        for i, module in enumerate(model.layers):
            for param in module.parameters():
                if i < freeze_up_to:
                    param.requires_grad = False
                    frozen += param.numel()
                else:
                    param.requires_grad = True
                    trainable += param.numel()
        for param in model.fc1.parameters():
            param.requires_grad = True
            trainable += param.numel()

    elif architecture == 'resnet':
        num_blocks = len(model.layers)
        freeze_blocks = max(1, int(num_blocks * freeze_ratio))

        for i, block in enumerate(model.layers):
            for param in block.parameters():
                if i < freeze_blocks:
                    param.requires_grad = False
                    frozen += param.numel()
                else:
                    param.requires_grad = True
                    trainable += param.numel()
        for param in model.final.parameters():
            param.requires_grad = True
            trainable += param.numel()

    elif architecture == 'inception':
        num_blocks = len(model.blocks) if hasattr(model, 'blocks') else 6
        freeze_blocks = max(1, int(num_blocks * freeze_ratio))

        if hasattr(model, 'blocks'):
            for i, block in enumerate(model.blocks):
                for param in block.parameters():
                    if i < freeze_blocks:
                        param.requires_grad = False
                        frozen += param.numel()
                    else:
                        param.requires_grad = True
                        trainable += param.numel()
        if hasattr(model, 'linear'):
            for param in model.linear.parameters():
                param.requires_grad = True
                trainable += param.numel()

    elif architecture == 'transformer':
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            num_layers = len(model.encoder.layers)
            freeze_layers_count = max(1, int(num_layers * freeze_ratio))

            for i, layer in enumerate(model.encoder.layers):
                for param in layer.parameters():
                    if i < freeze_layers_count:
                        param.requires_grad = False
                        frozen += param.numel()
                    else:
                        param.requires_grad = True
                        trainable += param.numel()

        # Freeze embeddings
        if hasattr(model, 'to_patch_embedding'):
            for param in model.to_patch_embedding.parameters():
                param.requires_grad = False
                frozen += param.numel()

        # Train classifier
        if hasattr(model, 'cls_layer'):
            for param in model.cls_layer.parameters():
                param.requires_grad = True
                trainable += param.numel()

    if verbose:
        print(f"\n{'='*70}")
        print(f"FREEZING EARLY LAYERS ({architecture.upper()})")
        print(f"{'='*70}")
        print(f"Frozen: {frozen:,} | Trainable: {trainable:,}")
        if frozen + trainable > 0:
            print(f"Trainable ratio: {100*trainable/(frozen+trainable):.1f}%")
        print(f"{'='*70}\n")

    return model


# ============================================================================
# Progressive Unfreezing
# ============================================================================

def get_unfreeze_schedule(model, architecture, num_epochs, unfreeze_every=5):
    """Create progressive unfreezing schedule"""
    if architecture == 'convnet':
        num_units = len(model.layers) // 3 if hasattr(model, 'layers') else 5
    elif architecture == 'resnet':
        num_units = len(model.layers) if hasattr(model, 'layers') else 3
    elif architecture == 'inception':
        num_units = len(model.blocks) if hasattr(model, 'blocks') else 6
    elif architecture == 'transformer':
        num_units = len(model.encoder.layers) if hasattr(
            model, 'encoder') else 4
    else:
        num_units = 4

    schedule = [min(1 + (e // unfreeze_every), num_units)
                for e in range(num_epochs)]
    return schedule


def apply_unfreeze(model, architecture, blocks_to_train):
    """Unfreeze last N blocks/layers"""
    for param in model.parameters():
        param.requires_grad = False

    if architecture == 'convnet':
        num_blocks = len(model.layers) // 3
        unfreeze_from = max(0, (num_blocks - blocks_to_train) * 3)
        for i, m in enumerate(model.layers):
            if i >= unfreeze_from:
                for p in m.parameters():
                    p.requires_grad = True
        for p in model.fc1.parameters():
            p.requires_grad = True

    elif architecture == 'resnet':
        unfreeze_from = max(0, len(model.layers) - blocks_to_train)
        for i, block in enumerate(model.layers):
            if i >= unfreeze_from:
                for p in block.parameters():
                    p.requires_grad = True
        for p in model.final.parameters():
            p.requires_grad = True

    elif architecture == 'inception' and hasattr(model, 'blocks'):
        unfreeze_from = max(0, len(model.blocks) - blocks_to_train)
        for i, block in enumerate(model.blocks):
            if i >= unfreeze_from:
                for p in block.parameters():
                    p.requires_grad = True
        if hasattr(model, 'linear'):
            for p in model.linear.parameters():
                p.requires_grad = True

    elif architecture == 'transformer':
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            unfreeze_from = max(0, len(model.encoder.layers) - blocks_to_train)
            for i, layer in enumerate(model.encoder.layers):
                if i >= unfreeze_from:
                    for p in layer.parameters():
                        p.requires_grad = True
        if hasattr(model, 'cls_layer'):
            for p in model.cls_layer.parameters():
                p.requires_grad = True


# ============================================================================
# Layer-wise Learning Rates
# ============================================================================

def get_layerwise_optimizer(model, architecture, config, verbose=True):
    """Architecture-aware layer-wise LR"""
    early, middle, head = [], [], []

    # Define keywords per architecture
    if architecture == 'convnet':
        classifier_kw = 'fc1'
        early_kw = ['layers.0', 'layers.1', 'layers.2',
                    'layers.3', 'layers.4', 'layers.5']
        middle_kw = ['layers.6', 'layers.7',
                     'layers.8', 'layers.9', 'layers.10']
    elif architecture == 'resnet':
        classifier_kw = 'final'
        early_kw = ['layers.0']
        middle_kw = ['layers.1', 'layers.2']
    elif architecture == 'inception':
        classifier_kw = 'linear'
        early_kw = ['blocks.0', 'blocks.1']
        middle_kw = ['blocks.2', 'blocks.3', 'blocks.4']
    elif architecture == 'transformer':
        classifier_kw = 'cls_layer'
        early_kw = ['patch_embedding',
                    'transformer.layers.0', 'transformer.layers.1']
        middle_kw = ['transformer.layers']
    else:
        classifier_kw, early_kw, middle_kw = 'classifier', [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if classifier_kw in name:
            head.append(
                {'params': [param], 'lr': config['head_lr'], 'name': name})
        elif any(k in name for k in early_kw):
            early.append(
                {'params': [param], 'lr': config['backbone_lr'], 'name': name})
        else:

            middle.append(
                {'params': [param], 'lr': config['middle_lr'], 'name': name})

    param_groups = early + middle + head

    if verbose:
        print(f"\n{'='*70}")
        print(f"{architecture.upper()} LAYER-WISE LR")
        print(f"{'='*70}")
        e_cnt = sum(sum(p.numel() for p in g['params']) for g in early)
        m_cnt = sum(sum(p.numel() for p in g['params']) for g in middle)
        h_cnt = sum(sum(p.numel() for p in g['params']) for g in head)
        print(f"Early   (LR {config['backbone_lr']:.2e}): {e_cnt:>10,} params")
        print(f"Middle  (LR {config['middle_lr']:.2e}): {m_cnt:>10,} params")
        print(f"Head    (LR {config['head_lr']:.2e}): {h_cnt:>10,} params")
        print(f"{'='*70}\n")

    return optim.AdamW(param_groups, betas=config['betas'], eps=config['eps'],
                       weight_decay=config['weight_decay'])


# ============================================================================
# Training Utilities
# ============================================================================

def get_cosine_schedule(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-7):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = (step - num_warmup_steps) / \
            max(1, num_training_steps - num_warmup_steps)
        return max(min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def train_epoch(model, loader, criterion, optimizer, scheduler, config, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    scaler = GradScaler() if config['use_amp'] else None

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(config['device']).float(
        ), labels.to(config['device']).long()

        if scaler:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            if config['gradient_clip'] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), config['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if config['gradient_clip'] > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), config['gradient_clip'])
            optimizer.step()

        optimizer.zero_grad()

        if scheduler and config['scheduler_step'] == 'batch':
            scheduler.step()

        total_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

        if batch_idx % config['log_interval'] == 0:
            pbar.set_postfix({
                'loss': f"{total_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

    return {'loss': total_loss/len(loader), 'accuracy': 100.0*correct/total}


@torch.no_grad()
def evaluate(model, loader, criterion, config, desc="Val"):
    if len(loader) == 0:
        return {'loss': 0, 'accuracy': 0, 'top3_accuracy': 0,
                'top5_accuracy': 0, 'f1_macro': 0, 'f1_weighted': 0}, [], [], []

    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    for inputs, labels in tqdm(loader, desc=desc, leave=False):
        inputs, labels = inputs.to(config['device']).float(
        ), labels.to(config['device']).long()
        outputs = model(inputs)
        total_loss += criterion(outputs, labels).item()

        probs = F.softmax(outputs, dim=1)
        all_preds.extend(outputs.max(1)[1].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds, all_labels, all_probs = np.array(
        all_preds), np.array(all_labels), np.array(all_probs)
    acc = 100.0 * accuracy_score(all_labels, all_preds)

    try:
        n_classes = all_probs.shape[1]
        top3 = 100.0 * \
            top_k_accuracy_score(all_labels, all_probs, k=min(
                3, n_classes), labels=np.arange(n_classes))
        top5 = 100.0 * \
            top_k_accuracy_score(all_labels, all_probs, k=min(
                5, n_classes), labels=np.arange(n_classes))
    except:
        top3 = top5 = acc

    return {
        'loss': total_loss/len(loader),
        'accuracy': acc,
        'top3_accuracy': top3,
        'top5_accuracy': top5,
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    }, all_preds, all_labels, all_probs


# ============================================================================
# Main Finetuning Function
# ============================================================================

def finetune_deep_model(
    data_path, model_name, pretrained_path, model_parameters_file,
    output_dir='results/finetuned',
    # Data
    split_per=0.7, seed=42, read_from_file=None, batch_size=32,
    # Finetuning strategy
    use_lora=False, lora_rank=16, lora_alpha=32.0,
    use_freezing=True, freeze_ratio=0.3,
    use_progressive_unfreeze=False, unfreeze_every=5,
    use_layer_wise_lr=True,
    backbone_lr=1e-5, middle_lr=5e-5, head_lr=5e-4,
    # Training
    num_epochs=30, warmup_epochs=3,
    weight_decay=0.05, gradient_clip=0.5,
    label_smoothing=0.1,
    # SWA
    use_swa=False, swa_start=20, swa_lr=5e-6,
    # Other
    early_stopping_patience=10, eval_model=True
):

    print("\n" + "="*80)
    print("UNIVERSAL MSAD MODEL FINETUNING")
    print("="*80)

    if not MSAD_AVAILABLE:
        raise ImportError("MSAD utilities not found")

    # Setup
    window_size = int(re.search(r'\d+', str(data_path)).group())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nWindow: {window_size} | Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(f"{output_dir}/weights", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{model_name}_{window_size}_{timestamp}"

    # Data loading
    print(f"\n{'='*80}")
    print("DATA LOADING")
    print(f"{'='*80}")

    train_set, val_set, test_set = create_splits(
        data_path, split_per, seed, read_from_file)
    training_data = TimeseriesDataset(data_path, fnames=train_set)
    val_data = TimeseriesDataset(data_path, fnames=val_set)
    test_data = TimeseriesDataset(data_path, fnames=test_set)

    print(
        f"Train: {len(training_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    class_weights = training_data.get_weights_subset(device)
    num_classes = len(class_weights)

    train_loader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model setup
    print(f"\n{'='*80}")
    print("MODEL SETUP")
    print(f"{'='*80}")

    model_parameters = json_file(model_parameters_file)
    if 'original_length' in model_parameters:
        model_parameters['original_length'] = window_size
    if 'timeseries_size' in model_parameters:
        model_parameters['timeseries_size'] = window_size
    if 'num_classes' in model_parameters:
        model_parameters['num_classes'] = num_classes

    model = deep_models[model_name.lower()](**model_parameters).to(device)

    # Load pretrained
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"\nLoading pretrained: {pretrained_path}")
        try:
            ckpt = torch.load(pretrained_path, map_location=device)
            if isinstance(ckpt, dict):
                state_dict = ckpt.get('model_state_dict') or ckpt.get(
                    'state_dict') or ckpt.get('model') or ckpt
            else:
                state_dict = ckpt
            model.load_state_dict(state_dict, strict=True)
            print("✓ Pretrained weights loaded")
        except Exception as e:
            print(f"⚠ Could not load weights: {e}")

    # Detect architecture
    architecture = detect_architecture(model)
    print(f"\nDetected architecture: {architecture.upper()}")

    # Apply finetuning strategy
    if use_lora and architecture == 'transformer':
        print(f"\n{'='*80}")
        print("APPLYING LoRA")
        print(f"{'='*80}")
        model, lora_layers = apply_lora_to_model(
            model, lora_rank, lora_alpha, verbose=True)
    elif use_freezing:
        model = freeze_layers(model, architecture, freeze_ratio, verbose=True)

    # Progressive unfreezing schedule
    unfreeze_schedule = None
    if use_progressive_unfreeze:
        unfreeze_schedule = get_unfreeze_schedule(
            model, architecture, num_epochs, unfreeze_every)
        print(
            f"\nProgressive unfreezing enabled (every {unfreeze_every} epochs)")

    # Training config
    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*80}")

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
        'scheduler_step': 'batch'
    }

    print(f"Epochs: {num_epochs} | Batch: {batch_size}")
    print(
        f"LR - Backbone: {backbone_lr:.2e}, Middle: {middle_lr:.2e}, Head: {head_lr:.2e}")
    print(f"Weight decay: {weight_decay} | Grad clip: {gradient_clip}")
    print(f"Label smoothing: {label_smoothing}")

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(
        device), label_smoothing=label_smoothing)

    # Optimizer
    if use_layer_wise_lr:
        optimizer = get_layerwise_optimizer(
            model, architecture, config, verbose=True)
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=head_lr, betas=config['betas'], eps=config['eps'], weight_decay=weight_decay
        )
        print(f"\nSingle LR: {head_lr:.2e}")

    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)

    # Early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience)

    # SWA
    swa_model = swa_scheduler = None
    if use_swa and swa_start < num_epochs:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        print(f"\nSWA enabled: starts epoch {swa_start}, LR={swa_lr:.2e}")

    # Training loop
    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}\n")

    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'val_top3_acc': [], 'val_top5_acc': [], 'val_f1_macro': [], 'learning_rates': []
    }

    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Progressive unfreezing
        if unfreeze_schedule:
            apply_unfreeze(model, architecture, unfreeze_schedule[epoch])
            # Recreate optimizer with new trainable params
            if use_layer_wise_lr:
                optimizer = get_layerwise_optimizer(
                    model, architecture, config, verbose=False)
            else:
                optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=head_lr, weight_decay=weight_decay
                )
            scheduler = get_cosine_schedule(
                optimizer, warmup_steps, total_steps)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer,
                                    scheduler if config['scheduler_step'] == 'batch' else None, config, epoch)

        # Validate
        val_metrics, _, _, _ = evaluate(
            model, val_loader, criterion, config, f"Epoch {epoch+1} [Val]")

        # Update scheduler (epoch-based)
        if scheduler and config['scheduler_step'] == 'epoch':
            scheduler.step()

        # SWA
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # Record
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_top3_acc'].append(val_metrics['top3_accuracy'])
        history['val_top5_acc'].append(val_metrics['top5_accuracy'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(
            f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, "
              f"Top-3: {val_metrics['top3_accuracy']:.2f}%, Top-5: {val_metrics['top5_accuracy']:.2f}%")

        # Save best
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'], 'config': config,
            }, f"{output_dir}/weights/{run_name}_best.pth")
            print(f"  ✓ Best model saved (Val Acc: {best_val_acc:.2f}%)")

        # Early stopping
        if early_stopping(val_metrics['accuracy'], epoch):
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            print(
                f"  Best: {early_stopping.best_score:.2f}% at epoch {early_stopping.best_epoch+1}")
            break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # SWA finalization
    if use_swa and swa_model:
        print("\nUpdating SWA batch norm...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        model = swa_model

    # Final evaluation
    print(f"\n{'='*80}")
    print("FINAL EVALUATION")
    print(f"{'='*80}\n")

    if eval_model and len(test_loader) > 0:
        test_metrics, test_preds, test_labels, _ = evaluate(
            model, test_loader, criterion, config, "Test")
        print(f"Test Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
        print(
            f"  Top-3: {test_metrics['top3_accuracy']:.2f}%, Top-5: {test_metrics['top5_accuracy']:.2f}%")
        print(
            f"  F1-Macro: {test_metrics['f1_macro']:.4f}, F1-Weighted: {test_metrics['f1_weighted']:.4f}")

        # Save results
        pd.DataFrame([test_metrics]).to_csv(
            f"{output_dir}/logs/{run_name}_test_results.csv", index=False)

        # Confusion matrix
        if len(test_labels) > 0:
            cm = confusion_matrix(test_labels, test_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
            plt.title('Test Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(
                f"{output_dir}/plots/{run_name}_confusion_matrix.png", dpi=150)
            plt.close()
    else:
        print("⚠ Test set empty, skipping evaluation")

    # Save history
    pd.DataFrame(history).to_csv(
        f"{output_dir}/logs/{run_name}_history.csv", index=False)

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Val Top-1')
    axes[0, 1].plot(history['val_top3_acc'], label='Val Top-3', linestyle='--')
    axes[0, 1].plot(history['val_top5_acc'], label='Val Top-5', linestyle=':')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history['learning_rates'])
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)

    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/{run_name}_training_curves.png", dpi=150)
    plt.close()

    print(f"\n{'='*80}")
    print(f"✓ Finetuning complete!")
    print(f"  Run: {run_name}")
    print(f"  Best model: {output_dir}/weights/{run_name}_best.pth")
    print(f"{'='*80}\n")

    return model, history


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Universal MSAD Finetuning')

    # Required
    parser.add_argument('-p', '--path', required=True, help='Dataset path')
    parser.add_argument('-m', '--model', required=True, help='Model name')
    parser.add_argument('-w', '--weights', required=True,
                        help='Pretrained weights')
    parser.add_argument('-pa', '--params', required=True,
                        help='Model params JSON')

    # Data
    parser.add_argument('-s', '--split', type=float, default=0.7)
    parser.add_argument('-se', '--seed', type=int, default=42)
    parser.add_argument('-f', '--file', default=None)
    parser.add_argument('-b', '--batch', type=int, default=32)

    # Finetuning strategy
    parser.add_argument('--use-lora', action='store_true',
                        help='LoRA (for Transformers)')
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=float, default=32.0)
    parser.add_argument('--use-freezing', action='store_true',
                        default=True, help='Freeze early layers')
    parser.add_argument('--freeze-ratio', type=float,
                        default=0.3, help='Fraction to freeze (0-1)')
    parser.add_argument('--progressive-unfreeze',
                        action='store_true', help='Progressive unfreezing')
    parser.add_argument('--unfreeze-every', type=int, default=5)
    parser.add_argument('--layer-wise-lr', action='store_true',
                        default=True, help='Layer-wise LR')
    parser.add_argument('--backbone-lr', type=float, default=1e-5)
    parser.add_argument('--middle-lr', type=float, default=5e-5)
    parser.add_argument('--head-lr', type=float, default=5e-4)

    # Training
    parser.add_argument('-ep', '--epochs', type=int, default=30)
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--gradient-clip', type=float, default=0.5)
    parser.add_argument('--label-smoothing', type=float, default=0.1)

    # SWA
    parser.add_argument('--use-swa', action='store_true')
    parser.add_argument('--swa-start', type=int, default=20)
    parser.add_argument('--swa-lr', type=float, default=5e-6)

    # Other
    parser.add_argument('--early-stopping', type=int, default=10)
    parser.add_argument('-o', '--output', default='results/finetuned')
    parser.add_argument('-e', '--eval-true', action='store_true')

    args = parser.parse_args()

    finetune_deep_model(
        data_path=args.path, model_name=args.model,
        pretrained_path=args.weights, model_parameters_file=args.params,
        output_dir=args.output,
        split_per=args.split, seed=args.seed, read_from_file=args.file,
        batch_size=args.batch,
        use_lora=args.use_lora, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        use_freezing=args.use_freezing, freeze_ratio=args.freeze_ratio,
        use_progressive_unfreeze=args.progressive_unfreeze,
        unfreeze_every=args.unfreeze_every,
        use_layer_wise_lr=args.layer_wise_lr,
        backbone_lr=args.backbone_lr, middle_lr=args.middle_lr, head_lr=args.head_lr,
        num_epochs=args.epochs, warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay, gradient_clip=args.gradient_clip,
        label_smoothing=args.label_smoothing,
        use_swa=args.use_swa, swa_start=args.swa_start, swa_lr=args.swa_lr,
        early_stopping_patience=args.early_stopping, eval_model=args.eval_true
    )
