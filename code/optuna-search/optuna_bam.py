"""
OPTUNA TUNING FOR BAM MODULE - ADJUSTED FOR YOUR SETUP
========================================================

Adjusted to work with your existing LLaMA-Factory setup
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import yaml
import subprocess
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ========================================
# CONFIGURATION
# ========================================

CONFIG_FILE = "train_bam_sampling.yaml"
MODULE_NAME = "BAM"
RESULT_DIR = "tuning_results_bam"
PLOT_DIR = "plots_bam"
N_TRIALS = 30

# Create directories
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Trial log
TRIAL_LOG = []


def load_base_config(config_file):
    """Load base configuration"""
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config


def run_training(trial_num, config):
    """Run single training trial"""
    
    # Create trial-specific output dir
    output_dir = f"outputs/bam_trial_{trial_num}"
    config["output_dir"] = output_dir
    
    # Create trial-specific config file
    trial_config_file = f"train_bam_trial_{trial_num}.yaml"
    
    with open(trial_config_file, "w") as f:
        yaml.dump(config, f)
    
    print(f"\n{'='*60}")
    print(f"Running Trial {trial_num}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    # Run training
    try:
        result = subprocess.run(
            ["llamafactory-cli", "train", trial_config_file],
            check=True,
            capture_output=True,
            text=True
        )
        print("Training completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Training failed for trial {trial_num}")
        print(f"Error: {e.stderr}")
        return None
    
    # Read results
    state_file = os.path.join(output_dir, "trainer_state.json")
    
    if not os.path.exists(state_file):
        print(f"Warning: trainer_state.json not found for trial {trial_num}")
        return None
    
    with open(state_file) as f:
        data = json.load(f)
    
    # Extract eval losses
    logs = data.get("log_history", [])
    eval_losses = [log["eval_loss"] for log in logs if "eval_loss" in log]
    
    if len(eval_losses) == 0:
        print(f"Warning: No eval_loss found for trial {trial_num}")
        return None
    
    # Return best eval loss
    best_eval_loss = min(eval_losses)
    final_eval_loss = eval_losses[-1]
    
    print(f"Trial {trial_num} Results:")
    print(f"  Best eval_loss: {best_eval_loss:.4f}")
    print(f"  Final eval_loss: {final_eval_loss:.4f}")
    
    return best_eval_loss


def objective(trial):
    
    # Load base config
    base_config = load_base_config(CONFIG_FILE)
    
    # ========================================
    # SEARCH SPACE - FOCUSED
    # ========================================
    
    # 1. LoRA Rank
    lora_rank = trial.suggest_categorical('lora_rank', [8, 16, 32])
    
    # 2. LoRA Alpha
    lora_alpha = lora_rank * 2
    
    # 3. Learning Rate (log scale)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 8e-5, log=True)
    
    # 4. Batch Size
    batch_size = trial.suggest_categorical('batch_size', [4, 8])
    
    # ========================================
    # FIXED PARAMETERS
    # ========================================
    
    # Gradient accumulation
    gradient_accumulation = 16 // batch_size
    
    # Training params
    num_epochs = 3
    lora_dropout = 0.05
    warmup_ratio = 0.03
    weight_decay = 0.0
    
    # ========================================
    # UPDATE CONFIG
    # ========================================
    
    config = base_config.copy()
    
    # LoRA params
    config["lora_rank"] = lora_rank
    config["lora_alpha"] = lora_alpha
    config["lora_dropout"] = lora_dropout
    
    # Training params
    config["learning_rate"] = learning_rate
    config["num_train_epochs"] = num_epochs
    config["per_device_train_batch_size"] = batch_size
    config["gradient_accumulation_steps"] = gradient_accumulation
    config["warmup_ratio"] = warmup_ratio
    config["weight_decay"] = weight_decay
    
    # Ensure eval and save strategies match
    config["eval_strategy"] = "steps"
    config["save_strategy"] = "steps"
    config["eval_steps"] = 100
    config["save_steps"] = 100
    config["load_best_model_at_end"] = True
    config["metric_for_best_model"] = "eval_loss"
    config["greater_is_better"] = False
    
    # ========================================
    # RUN TRAINING
    # ========================================
    
    eval_loss = run_training(trial.number, config)
    
    if eval_loss is None:
        # Trial failed
        raise optuna.TrialPruned()
    
    # ========================================
    # LOG TRIAL
    # ========================================
    
    trial_info = {
        "trial": trial.number,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch": batch_size * gradient_accumulation,
        "eval_loss": eval_loss,
        "timestamp": datetime.now().isoformat()
    }
    
    TRIAL_LOG.append(trial_info)
    
    # Save incrementally
    df = pd.DataFrame(TRIAL_LOG)
    df.to_csv(f"{RESULT_DIR}/trials_progress.csv", index=False)
    
    print(f"\nTrial {trial.number} Summary:")
    print(f"  Rank: {lora_rank}, Alpha: {lora_alpha}")
    print(f"  LR: {learning_rate:.2e}")
    print(f"  Batch: {batch_size} × {gradient_accumulation} = {batch_size*gradient_accumulation}")
    print(f"  Eval Loss: {eval_loss:.4f}")
    
    return eval_loss


def main():
    """Main optimization function"""
    
    print("="*80)
    print(f"OPTUNA HYPERPARAMETER TUNING - {MODULE_NAME} MODULE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Config file: {CONFIG_FILE}")
    print(f"  Number of trials: {N_TRIALS}")
    print(f"  Results directory: {RESULT_DIR}")
    
    # ========================================
    # CREATE OPTUNA STUDY
    # ========================================
    
    study = optuna.create_study(
        study_name=f"{MODULE_NAME}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        direction="minimize",
        
        sampler=TPESampler(
            seed=42,
            n_startup_trials=5
        ),
        
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=100,
            interval_steps=50
        ),
        
        storage=f"sqlite:///{RESULT_DIR}/optuna_study.db",
        load_if_exists=True
    )
    
    # ========================================
    # RUN OPTIMIZATION
    # ========================================
    
    print(f"\nStarting optimization...")
    print(f"This may take 3-5 days.\n")
    
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
        catch=(Exception,)
    )
    
    # ========================================
    # RESULTS ANALYSIS
    # ========================================
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    
    print(f"\nNumber of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    # Best trial
    print("\n" + "="*80)
    print("BEST TRIAL")
    print("="*80)
    
    best_trial = study.best_trial
    print(f"\nTrial number: {best_trial.number}")
    print(f"Eval Loss: {best_trial.value:.4f}")
    print(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    best_rank = best_trial.params['lora_rank']
    best_batch = best_trial.params['batch_size']
    print(f"\nDerived Parameters:")
    print(f"  lora_alpha: {best_rank * 2}")
    print(f"  gradient_accumulation: {16 // best_batch}")
    print(f"  effective_batch_size: 16")
    
    # Save best params
    best_params = {
        'trial_number': best_trial.number,
        'eval_loss': best_trial.value,
        'params': best_trial.params,
        'full_config': {
            'lora_rank': best_trial.params['lora_rank'],
            'lora_alpha': best_rank * 2,
            'learning_rate': best_trial.params['learning_rate'],
            'per_device_train_batch_size': best_trial.params['batch_size'],
            'gradient_accumulation_steps': 16 // best_batch,
            'num_train_epochs': 3,
            'lora_dropout': 0.05,
            'warmup_ratio': 0.03,
            'weight_decay': 0.0
        }
    }
    
    with open(f"{RESULT_DIR}/best_params.json", 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\nBest params saved to: {RESULT_DIR}/best_params.json")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    
    df = pd.DataFrame(TRIAL_LOG)
    df_sorted = df.sort_values('eval_loss')
    
    print("\n" + "="*80)
    print("TOP 10 TRIALS")
    print("="*80)
    print(df_sorted[['trial', 'lora_rank', 'learning_rate', 'batch_size', 'eval_loss']].head(10))
    
    df_sorted.to_csv(f"{RESULT_DIR}/all_trials_sorted.csv", index=False)
    print(f"\nFull results saved to: {RESULT_DIR}/all_trials_sorted.csv")
    
    # ========================================
    # VISUALIZATIONS
    # ========================================
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f"{PLOT_DIR}/optimization_history.html")
        print(f"✓ Optimization history saved")
    except Exception as e:
        print(f"✗ Could not generate optimization history: {e}")
    
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f"{PLOT_DIR}/param_importances.html")
        print(f"✓ Parameter importances saved")
    except Exception as e:
        print(f"✗ Could not generate param importances: {e}")
    
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(f"{PLOT_DIR}/parallel_coordinate.html")
        print(f"✓ Parallel coordinate saved")
    except Exception as e:
        print(f"✗ Could not generate parallel coordinate: {e}")
    
    # Custom plots
    plt.figure(figsize=(10, 6))
    plt.scatter(df['learning_rate'], df['eval_loss'], alpha=0.6, s=100)
    plt.xlabel('Learning Rate')
    plt.ylabel('Eval Loss')
    plt.title(f'{MODULE_NAME}: Learning Rate vs Eval Loss')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOT_DIR}/lr_vs_loss.png", dpi=150, bbox_inches='tight')
    print(f"✓ LR scatter plot saved")
    
    plt.figure(figsize=(10, 6))
    for rank in sorted(df['lora_rank'].unique()):
        mask = df['lora_rank'] == rank
        plt.scatter(
            df[mask]['trial'],
            df[mask]['eval_loss'],
            label=f'Rank {rank}',
            s=100,
            alpha=0.6
        )
    plt.xlabel('Trial Number')
    plt.ylabel('Eval Loss')
    plt.title(f'{MODULE_NAME}: LoRA Rank Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOT_DIR}/rank_comparison.png", dpi=150, bbox_inches='tight')
    print(f"✓ Rank comparison saved")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Review: {RESULT_DIR}/best_params.json")
    print(f"2. Check plots in: {PLOT_DIR}/")
    print(f"3. Train final model with best config")
    print("="*80)


if __name__ == "__main__":
    main()