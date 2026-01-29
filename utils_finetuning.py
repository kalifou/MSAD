#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################import numpy as np
############################
# Universal Finetuning Script for All MSAD Architectures
# Supports: ConvNet, ResNet, InceptionTime, Transformer (SIT)
########################################################################
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings('ignore')




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
