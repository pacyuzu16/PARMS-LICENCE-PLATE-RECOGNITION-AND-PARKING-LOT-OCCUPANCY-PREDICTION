#!/usr/bin/env python3
"""
PARMS - Parking Management System
Simple parking occupancy prediction with data preprocessing and model training

Usage:
    python parms.py                    # Run complete pipeline
    python parms.py --preprocess       # Data preprocessing only
    python parms.py --train            # Model training only
    python parms.py --predict IMAGE    # Predict on single image
    python parms.py --demo             # Run interactive demo
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from data_processor import DataProcessor
from model_trainer import ModelTrainer
from config import Config


class PARMS:
    """Main PARMS application class."""
    
    def __init__(self):
        self.config = Config()
        self.data_processor = DataProcessor(self.config)
        self.model_trainer = ModelTrainer(self.config)
        
    def run_complete_pipeline(self):
        """Run the complete PARMS pipeline."""
        print("PARMS - Parking Management System")
        print("=" * 50)
        print("Running complete pipeline: Data -> Model -> Analysis")
        print()
        
        # Step 1: Data Processing
        if not os.path.exists(self.config.processed_data_path):
            print("Step 1: Processing parking data...")
            self.data_processor.process_all()
        else:
            print(f"Found processed data: {self.config.processed_data_path}")
        
        # Step 2: Model Training
        print("Step 2: Training parking model...")
        model, metrics = self.model_trainer.train_and_evaluate()
        
        # Step 3: Results Summary
        self.print_results_summary(metrics)
        
        return model, metrics
    
    def preprocess_only(self):
        """Run only data preprocessing."""
        print("PARMS - Data Preprocessing Only")
        print("=" * 40)
        self.data_processor.process_all()
        
    def train_only(self):
        """Run only model training."""
        print("PARMS - Model Training Only")
        print("=" * 40)
        
        if not os.path.exists(self.config.processed_data_path):
            print("[ERROR] Processed data not found. Run preprocessing first:")
            print("   python parms.py --preprocess")
            return None
            
        return self.model_trainer.train_and_evaluate()
    
    def predict_single(self, image_path):
        """Predict parking occupancy for a single image."""
        print(f"[INFO] PARMS - Predicting: {image_path}")
        print("=" * 40)
        
        # Load trained model
        model_path = self.config.model_save_path
        if not os.path.exists(model_path):
            print("[ERROR] Trained model not found. Train model first:")
            print("   python parms.py --train")
            return None
            
        # TODO: Implement single image prediction
        print("WARNING: Single image prediction not yet implemented")
        print("   This would require image processing and feature extraction")
        
    def run_demo(self):
        """Run interactive demonstration."""
        print("PARMS - Interactive Demo")
        print("=" * 30)
        
        try:
            model, metrics = self.run_complete_pipeline()
            
            print("\nDemo Results:")
            print(f"   Model Accuracy: {metrics.get('accuracy', 0):.2%}")
            print(f"   Precision: {metrics.get('precision', 0):.2%}")
            print(f"   Recall: {metrics.get('recall', 0):.2%}")
            print(f"   F1-Score: {metrics.get('f1', 0):.2%}")
            
            print("\nDemo completed successfully!")
            
        except Exception as e:
            print(f"ERROR: Demo failed: {e}")
    
    def print_results_summary(self, metrics):
        """Print a nice summary of results."""
        print("\nPARMS PIPELINE COMPLETED!")
        print("=" * 40)
        
        print("Model Performance:")
        print(f"   Accuracy: {metrics.get('accuracy', 0):.2%}")
        print(f"   Precision: {metrics.get('precision', 0):.2%}")
        print(f"   Recall: {metrics.get('recall', 0):.2%}")
        print(f"   F1-Score: {metrics.get('f1', 0):.2%}")
        
        print("\nGenerated Files:")
        if os.path.exists(self.config.processed_data_path):
            print(f"   Processed data: {self.config.processed_data_path}")
        if os.path.exists(self.config.model_save_path):
            print(f"   Trained model: {self.config.model_save_path}")
        if os.path.exists("Generated Figures"):
            print(f"   Analysis plots: Generated Figures/")
        
        print("\nYour parking management system is ready!")


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description="PARMS - Parking Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--preprocess", 
        action="store_true",
        help="Run data preprocessing only"
    )
    
    parser.add_argument(
        "--train", 
        action="store_true",
        help="Run model training only"
    )
    
    parser.add_argument(
        "--predict",
        metavar="IMAGE_PATH",
        help="Predict parking occupancy for single image"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true", 
        help="Run interactive demonstration"
    )
    
    args = parser.parse_args()
    
    # Create PARMS instance
    parms = PARMS()
    
    try:
        # Route to appropriate function
        if args.preprocess:
            parms.preprocess_only()
        elif args.train:
            parms.train_only()
        elif args.predict:
            parms.predict_single(args.predict)
        elif args.demo:
            parms.run_demo()
        else:
            # Default: run complete pipeline
            parms.run_complete_pipeline()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()