"""Command Line Interface for LEAP framework."""

import click
import torch
import os
import sys
from pathlib import Path
from typing import Optional

from .framework import LEAPFramework
from .config import LEAPConfig
from .utils.data_utils import create_leap_dataloaders
from .utils.common import set_seed, setup_logging, get_device, create_output_dir


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """LEAP: Learning Expert Adaptation & Pruning for MoE Language Models."""
    pass


@cli.command()
@click.option(
    "--config", 
    "-c", 
    type=click.Path(exists=True), 
    required=True,
    help="Path to configuration YAML file"
)
@click.option(
    "--model-path", 
    "-m", 
    type=click.Path(exists=True),
    help="Path to pre-trained model"
)
@click.option(
    "--output-dir", 
    "-o", 
    type=click.Path(),
    default="./outputs",
    help="Output directory"
)
@click.option(
    "--debug", 
    is_flag=True,
    help="Enable debug mode with reduced data"
)
def prune_model(config: str, model_path: Optional[str], output_dir: str, debug: bool):
    """Run expert pruning phase only."""
    
    click.echo("🔥 LEAP Expert Pruning")
    click.echo("=" * 40)
    
    # Load configuration
    leap_config = LEAPConfig.from_yaml(config)
    
    # Setup
    device = get_device()
    set_seed(leap_config.seed)
    output_dir = create_output_dir(output_dir, f"{leap_config.experiment_name}_pruning")
    
    setup_logging(log_level="DEBUG" if debug else "INFO")
    
    # Debug adjustments
    if debug:
        leap_config.pruning.meta_episodes = 10
    
    # Create data loaders
    click.echo("📊 Loading data...")
    dataloaders = create_leap_dataloaders(
        task=leap_config.task,
        tokenizer_name="microsoft/DialoGPT-medium",
        batch_size=leap_config.training.batch_size,
        max_length=leap_config.training.max_length,
        num_train_samples=1000 if debug else None,
        num_val_samples=200 if debug else None,
    )
    
    # Initialize LEAP
    leap = LEAPFramework(leap_config, device)
    model = leap.load_model(model_path)
    
    click.echo(f"🏗️  Model: {model.__class__.__name__}")
    click.echo(f"🎯 Target experts: {leap_config.pruning.target_experts}/{leap_config.model.num_experts}")
    
    # Run pruning
    try:
        pruned_experts = leap.run_pruning_phase(
            dataloaders["train"], 
            dataloaders["val"],
            save_path=os.path.join(output_dir, "pruned_model.pt")
        )
        
        click.echo("✅ Pruning completed!")
        click.echo(f"Selected experts: {pruned_experts}")
        click.echo(f"💾 Model saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--config", 
    "-c", 
    type=click.Path(exists=True), 
    required=True,
    help="Path to configuration YAML file"
)
@click.option(
    "--model-path", 
    "-m", 
    type=click.Path(exists=True),
    required=True,
    help="Path to pruned model"
)
@click.option(
    "--output-dir", 
    "-o", 
    type=click.Path(),
    default="./outputs",
    help="Output directory"
)
@click.option(
    "--debug", 
    is_flag=True,
    help="Enable debug mode"
)
def adapt_routing(config: str, model_path: str, output_dir: str, debug: bool):
    """Run routing adaptation phase only."""
    
    click.echo("🧭 LEAP Routing Adaptation")
    click.echo("=" * 40)
    
    # Load configuration
    leap_config = LEAPConfig.from_yaml(config)
    
    # Setup
    device = get_device()
    set_seed(leap_config.seed)
    output_dir = create_output_dir(output_dir, f"{leap_config.experiment_name}_routing")
    
    setup_logging(log_level="DEBUG" if debug else "INFO")
    
    # Debug adjustments
    if debug:
        leap_config.training.total_steps = 100
    
    # Create data loaders
    click.echo("📊 Loading data...")
    dataloaders = create_leap_dataloaders(
        task=leap_config.task,
        tokenizer_name="microsoft/DialoGPT-medium",
        batch_size=leap_config.training.batch_size,
        max_length=leap_config.training.max_length,
        num_train_samples=1000 if debug else None,
        num_val_samples=200 if debug else None,
    )
    
    # Initialize LEAP and load pruned model
    leap = LEAPFramework(leap_config, device)
    leap.load_full_checkpoint(model_path)
    
    if not leap.is_pruned:
        click.echo("❌ Model must be pruned first. Run 'leap-prune' command.", err=True)
        sys.exit(1)
    
    click.echo(f"🏗️  Loaded pruned model with {len(leap.pruned_expert_indices)} experts")
    
    # Run routing adaptation
    try:
        results = leap.run_routing_phase(
            dataloaders["train"], 
            dataloaders["val"],
            save_path=os.path.join(output_dir, "adapted_model.pt")
        )
        
        click.echo("✅ Routing adaptation completed!")
        click.echo(f"Final performance: {results['final_performance']:.3f}")
        click.echo(f"💾 Model saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--config", 
    "-c", 
    type=click.Path(exists=True), 
    required=True,
    help="Path to configuration YAML file"
)
@click.option(
    "--model-path", 
    "-m", 
    type=click.Path(exists=True),
    help="Path to pre-trained model (optional)"
)
@click.option(
    "--output-dir", 
    "-o", 
    type=click.Path(),
    default="./outputs",
    help="Output directory"
)
@click.option(
    "--resume-from", 
    type=click.Path(exists=True),
    help="Resume from checkpoint"
)
@click.option(
    "--debug", 
    is_flag=True,
    help="Enable debug mode"
)
def train_model(config: str, model_path: Optional[str], output_dir: str, resume_from: Optional[str], debug: bool):
    """Run complete LEAP training pipeline."""
    
    click.echo("🚀 LEAP Complete Training")
    click.echo("=" * 40)
    
    # Load configuration
    leap_config = LEAPConfig.from_yaml(config)
    
    # Setup
    device = get_device()
    set_seed(leap_config.seed)
    output_dir = create_output_dir(output_dir, leap_config.experiment_name)
    
    setup_logging(log_level="DEBUG" if debug else "INFO")
    
    # Debug adjustments
    if debug:
        leap_config.pruning.meta_episodes = 10
        leap_config.training.total_steps = 100
    
    # Create data loaders
    click.echo("📊 Loading data...")
    dataloaders = create_leap_dataloaders(
        task=leap_config.task,
        tokenizer_name="microsoft/DialoGPT-medium",
        batch_size=leap_config.training.batch_size,
        max_length=leap_config.training.max_length,
        num_train_samples=1000 if debug else None,
        num_val_samples=200 if debug else None,
    )
    
    # Initialize LEAP
    leap = LEAPFramework(leap_config, device)
    
    if resume_from:
        click.echo(f"📂 Resuming from: {resume_from}")
        leap.load_full_checkpoint(resume_from)
    else:
        model = leap.load_model(model_path)
        click.echo(f"🏗️  Model: {model.__class__.__name__}")
    
    # Run optimization
    try:
        results = leap.optimize(
            train_data=dataloaders["train"],
            val_data=dataloaders["val"],
            save_path=os.path.join(output_dir, "leap_model")
        )
        
        click.echo("✅ Training completed!")
        
        # Show results
        model_info = results["model_info"]
        performance = results["evaluation_results"]["performance"]
        
        click.echo(f"Compression: {model_info['original_experts']} → {model_info['pruned_experts']} experts")
        click.echo(f"Performance: {performance.get(leap_config.metric, 0):.3f}")
        click.echo(f"💾 Results saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option(
    "--model-path", 
    "-m", 
    type=click.Path(exists=True),
    required=True,
    help="Path to model checkpoint"
)
@click.option(
    "--config", 
    "-c", 
    type=click.Path(exists=True),
    help="Path to configuration file (optional)"
)
@click.option(
    "--data-path", 
    "-d", 
    type=click.Path(exists=True),
    help="Path to evaluation data"
)
@click.option(
    "--output-file", 
    "-o", 
    type=click.Path(),
    default="evaluation_results.json",
    help="Output file for results"
)
@click.option(
    "--batch-size", 
    "-b", 
    type=int,
    default=8,
    help="Batch size for evaluation"
)
def evaluate_model(model_path: str, config: Optional[str], data_path: Optional[str], output_file: str, batch_size: int):
    """Evaluate a LEAP model."""
    
    click.echo("📊 LEAP Model Evaluation")
    click.echo("=" * 40)
    
    # Load configuration
    if config:
        leap_config = LEAPConfig.from_yaml(config)
    else:
        # Try to infer from model checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        if "config" in checkpoint:
            leap_config = checkpoint["config"]
        else:
            click.echo("❌ No configuration found. Please provide --config", err=True)
            sys.exit(1)
    
    # Setup
    device = get_device()
    set_seed(leap_config.seed)
    setup_logging()
    
    # Create data loader
    if data_path:
        # Load custom data (implementation would depend on format)
        click.echo("📁 Loading custom evaluation data...")
        # Placeholder - in practice, you'd implement custom data loading
        dataloaders = create_leap_dataloaders(
            task=leap_config.task,
            tokenizer_name="microsoft/DialoGPT-medium",
            batch_size=batch_size,
            max_length=leap_config.training.max_length,
            num_val_samples=500,
        )
        test_dataloader = dataloaders["val"]
    else:
        # Use default test data
        click.echo("📊 Loading default test data...")
        dataloaders = create_leap_dataloaders(
            task=leap_config.task,
            tokenizer_name="microsoft/DialoGPT-medium",
            batch_size=batch_size,
            max_length=leap_config.training.max_length,
            num_val_samples=1000,
        )
        test_dataloader = dataloaders["val"]
    
    # Initialize LEAP and load model
    leap = LEAPFramework(leap_config, device)
    leap.load_full_checkpoint(model_path)
    
    click.echo(f"🏗️  Model: {leap.model.__class__.__name__}")
    if leap.is_pruned:
        click.echo(f"🎯 Active experts: {len(leap.pruned_expert_indices)}/{leap.config.model.num_experts}")
    
    # Run evaluation
    try:
        results = leap.evaluate(leap.model, test_dataloader, leap.pruned_expert_indices)
        
        click.echo("✅ Evaluation completed!")
        click.echo("=" * 40)
        
        # Display key metrics
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                if "time" in metric.lower():
                    click.echo(f"{metric}: {value:.2f} ms")
                elif "accuracy" in metric.lower() or "rouge" in metric.lower():
                    click.echo(f"{metric}: {value:.3f}")
                elif "perplexity" in metric.lower():
                    click.echo(f"{metric}: {value:.2f}")
                else:
                    click.echo(f"{metric}: {value}")
        
        # Save results
        leap.evaluator.save_evaluation_results(results, output_file)
        click.echo(f"💾 Results saved to: {output_file}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--model-type", 
    type=click.Choice(["llama_maverick", "qwen3_235b"]),
    default="llama_maverick",
    help="Model type"
)
@click.option(
    "--task", 
    type=click.Choice(["code_generation", "reasoning", "summarization"]),
    default="code_generation",
    help="Task type"
)
@click.option(
    "--output-file", 
    "-o", 
    default="config.yaml",
    help="Output configuration file"
)
def generate_config(model_type: str, task: str, output_file: str):
    """Generate a configuration file template."""
    
    click.echo(f"📝 Generating configuration for {model_type} on {task}")
    
    # Create configuration
    config = LEAPConfig()
    
    # Set model configuration
    from .config import get_model_config, get_task_config
    config.model = get_model_config(model_type)
    config.task = task
    config.metric = get_task_config(task)["metric"]
    
    # Adjust pruning targets based on task
    task_config = get_task_config(task)
    config.pruning.target_experts = task_config["target_experts"]
    
    # Save configuration
    config.to_yaml(output_file)
    
    click.echo(f"✅ Configuration saved to: {output_file}")
    click.echo("💡 Edit the file to customize settings before training")


if __name__ == "__main__":
    cli()
