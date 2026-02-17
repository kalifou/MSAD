
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import copy
from datetime import datetime


from utils.finetuning import apply_lora_to_model, detect_architecture, \
    evaluate, freeze_layers, get_cosine_schedule, get_unfreeze_schedule, \
        apply_unfreeze, train_epoch, EarlyStopping, get_layerwise_optimizer
        
try:
    from utils.timeseries_dataset import create_splits, TimeseriesDataset
    from utils.train_deep_model_utils import json_file
    from utils.config import deep_models
    MSAD_AVAILABLE = True
except ImportError:
    print("ERROR: MSAD utils not found")
    MSAD_AVAILABLE = False

# ============================================================================
# Main Finetuning Function
# ============================================================================

def finetune_deep_model(
    data_path, model_name, pretrained_path, model_parameters_file,
    output_dir='results/finetuned',
    window_size=None,
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
    if window_size is None:
        raise ValueError("window_size must be provided explicitly.")
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
    
    import ipdb
    #ipdb.set_trace(context=55)
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

    #ipdb.set_trace(context=25)
    model_parameters = json_file(model_parameters_file)
    if 'original_length' in model_parameters:
        model_parameters['original_length'] = window_size
    if 'timeseries_size' in model_parameters:
        model_parameters['timeseries_size'] = window_size
    #if 'num_classes' in model_parameters:
    #    model_parameters['num_classes'] = num_classes

    
    #ipdb.set_trace(context=25)
    model = deep_models[model_name.lower()](**model_parameters).to(device)
    #ipdb.set_trace(context=25)
    
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
            #ipdb.set_trace(context=25)
            
            model.load_state_dict(state_dict, strict=True)
            print("✓ Pretrained weights loaded")
        except Exception as e:
            print(f"⚠ Could not load weights: {e}")
    
    #ipdb.set_trace(context=25)  
    if model_name == "inception_time":
        model.linear = nn.Linear(in_features=model.linear.in_features, out_features=7, bias=True)
    elif model_name == "convnet":
        model.fc1[0] = nn.Linear(model.fc1[0].in_features, out_features=7, bias=True)
    #ipdb.set_trace(context=25)
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

        #ipdb.set_trace(context=55)
        # Save best
        if val_metrics['accuracy'] > best_val_acc or val_set == list():
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
    parser.add_argument('-ws', '--window-size', type=int, required=True,
                        help='Length of the time series window (e.g. 128, 256)')

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
        window_size=args.window_size,
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
