#!/usr/bin/env python3
"""Complete LEAP pipeline example with all phases."""

import argparse
import os
import torch
from pathlib import Path

from leap import LEAPFramework
from leap.config import LEAPConfig
from leap.utils.data_utils import create_leap_dataloaders
from leap.utils.common import set_seed, setup_logging, get_device, create_output_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run complete LEAP optimization pipeline")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pre-trained model (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="leap_experiment",
        help="Name of the experiment"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation (no training)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with reduced data"
    )
    
    return parser.parse_args()


def main():
    """Run complete LEAP pipeline."""
    
    args = parse_args()
    
    # Load configuration
    config = LEAPConfig.from_yaml(args.config)
    
    # Override with command line arguments
    if args.experiment_name != "leap_experiment":
        config.experiment_name = args.experiment_name
    
    # Setup logging and output directory
    output_dir = create_output_dir(args.output_dir, config.experiment_name)
    config.training.output_dir = output_dir
    
    log_file = os.path.join(output_dir, "leap.log")
    setup_logging(log_level="DEBUG" if args.debug else "INFO", log_file=log_file)
    
    # Set device and seed
    device = get_device()
    set_seed(config.seed)
    
    print("🚀 LEAP: Learning Expert Adaptation & Pruning")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Task: {config.task}")
    print(f"🏗️  Model: {config.model.model_type}")
    print(f"🎲 Seed: {config.seed}")
    print(f"💾 Device: {device}")
    
    # Debug mode adjustments
    if args.debug:
        print("\n🐛 Debug mode enabled - reducing data and iterations")
        config.pruning.meta_episodes = 10
        config.training.total_steps = 100
        config.training.eval_steps = 20
        config.training.save_steps = 50
    
    # Create data loaders
    print("\n📊 Loading datasets...")
    
    num_train_samples = 1000 if args.debug else None
    num_val_samples = 200 if args.debug else None
    
    dataloaders = create_leap_dataloaders(
        task=config.task,
        tokenizer_name="microsoft/DialoGPT-medium",  # Placeholder
        batch_size=config.training.batch_size,
        max_length=config.training.max_length,
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
        num_workers=4 if not args.debug else 0
    )
    
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]
    
    print(f"✅ Data loaded:")
    print(f"   Training samples: {len(train_dataloader.dataset):,}")
    print(f"   Validation samples: {len(val_dataloader.dataset):,}")
    print(f"   Batch size: {config.training.batch_size}")
    
    # Initialize LEAP framework
    print(f"\n🧠 Initializing LEAP framework...")
    leap = LEAPFramework(config, device)
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"📂 Resuming from checkpoint: {args.resume_from}")
        leap.load_full_checkpoint(args.resume_from)
    else:
        # Load model
        print(f"🏗️  Loading model...")
        model = leap.load_model(args.model_path)
        
        # Print model statistics
        model_stats = leap.get_model_stats()
        print(f"✅ Model loaded:")
        print(f"   Type: {model_stats['model_type']}")
        print(f"   Total parameters: {model_stats['total_parameters']:,}")
        print(f"   Total experts: {model_stats['total_experts']}")
        print(f"   Target experts: {config.pruning.target_experts}")
    
    # Save configuration
    config_save_path = os.path.join(output_dir, "config.yaml")
    config.to_yaml(config_save_path)
    print(f"💾 Configuration saved to: {config_save_path}")
    
    if args.eval_only:
        print("\n📊 Running evaluation only...")
        
        # Create test dataloader (using validation for demo)
        test_results = leap.evaluate(leap.model, val_dataloader)
        
        print("📋 Evaluation Results:")
        for metric, value in test_results.items():
            if isinstance(value, (int, float)):
                print(f"   {metric}: {value:.4f}")
        
        # Save results
        results_path = os.path.join(output_dir, "evaluation_results.json")
        leap.evaluator.save_evaluation_results(test_results, results_path)
        print(f"💾 Results saved to: {results_path}")
        
    else:
        print("\n🔥 Starting LEAP optimization pipeline...")
        
        try:
            # Run optimization
            results = leap.optimize(
                train_data=train_dataloader,
                val_data=val_dataloader,
                save_path=os.path.join(output_dir, "leap_model")
            )
            
            # Print results
            print("\n✅ LEAP optimization completed successfully!")
            print("=" * 60)
            
            # Model compression results
            model_info = results["model_info"]
            print(f"🎯 Compression Results:")
            print(f"   Original experts: {model_info['original_experts']}")
            print(f"   Pruned experts: {model_info['pruned_experts']}")
            print(f"   Compression ratio: {model_info['compression_ratio']:.1%}")
            print(f"   Parameter reduction: {1 - model_info['compression_ratio']:.1%}")
            
            # Performance results
            eval_results = results["evaluation_results"]
            performance = eval_results["performance"]
            efficiency = eval_results["efficiency"]
            
            print(f"\n📊 Performance Results:")
            print(f"   Final {config.metric}: {performance.get(config.metric, 0):.4f}")
            print(f"   Perplexity: {performance.get('perplexity', 0):.2f}")
            print(f"   Inference time: {efficiency.get('inference_time_ms', 0):.1f} ms")
            print(f"   Tokens/second: {efficiency.get('tokens_per_second', 0):.1f}")
            print(f"   Memory usage: {efficiency.get('memory_allocated_gb', 0):.1f} GB")
            
            # Expert utilization
            utilization = eval_results["utilization"]
            if "per_layer_stats" in utilization:
                avg_active = sum(
                    stats["active_experts"] 
                    for stats in utilization["per_layer_stats"].values()
                ) / len(utilization["per_layer_stats"])
                print(f"   Avg active experts per layer: {avg_active:.1f}")
            
            # Save detailed results
            results_path = os.path.join(output_dir, "leap_results.json")
            leap.evaluator.save_evaluation_results(results, results_path)
            print(f"\n💾 Detailed results saved to: {results_path}")
            
            # Save final checkpoint
            checkpoint_path = os.path.join(output_dir, "final_checkpoint.pt")
            leap.save_full_checkpoint(checkpoint_path)
            print(f"💾 Final checkpoint saved to: {checkpoint_path}")
            
        except Exception as e:
            print(f"\n❌ Error during optimization: {e}")
            print("💡 Tips:")
            print("   - Ensure you have enough GPU memory")
            print("   - Try reducing batch size or sequence length")
            print("   - Use --debug flag for testing")
            print("   - Provide pre-trained model weights for full functionality")
            
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 LEAP pipeline completed!")
    print(f"📁 All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
