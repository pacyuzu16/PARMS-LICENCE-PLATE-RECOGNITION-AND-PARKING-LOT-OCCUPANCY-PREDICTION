"""
PARMS Configuration
Central configuration management for the PARMS parking system.
"""

import os
from pathlib import Path


class Config:
    """Configuration class for PARMS system."""
    
    def __init__(self):
        # Project paths
        self.project_root = Path(__file__).parent
        self.data_root = self.project_root / "data" / "pklot_reduced"
        
        # Data files
        self.processed_data_path = "data/processed_pklot_parms_coco.csv"
        self.model_save_path = "parms_model.pkl"
        
        # Processing settings
        self.max_samples = None   # Use full dataset
        self.test_size = 0.2      # Train/test split ratio
        self.random_seed = 42     # For reproducible results
        
        # Model settings
        self.model_type = "random_forest"
        self.n_estimators = 100
        self.max_depth = 10
        self.min_samples_split = 5
        
        # Output settings
        self.figures_dir = "Generated Figures"
        self.reports_dir = "reports"
        self.verbose = True
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        dirs_to_create = [
            "data",
            self.figures_dir,
            self.reports_dir
        ]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
    
    def get_data_info(self):
        """Get information about data availability."""
        info = {
            "data_root_exists": self.data_root.exists(),
            "processed_data_exists": os.path.exists(self.processed_data_path),
            "model_exists": os.path.exists(self.model_save_path),
        }
        
        # Check for COCO splits
        if self.data_root.exists():
            for split in ["train", "valid", "test"]:
                coco_file = self.data_root / split / "_annotations.coco.json"
                info[f"{split}_coco_exists"] = coco_file.exists()
        
        return info
    
    def print_status(self):
        """Print current configuration status."""
        print("PARMS Configuration")
        print("-" * 30)
        print(f"Project Root: {self.project_root}")
        print(f"Data Root: {self.data_root}")
        print(f"Max Samples: {self.max_samples if self.max_samples else 'All (no limit)'}")
        print(f"Model Type: {self.model_type}")
        
        print("\nFile Status:")
        info = self.get_data_info()
        for key, exists in info.items():
            status = "[OK]" if exists else "[MISSING]"
            print(f"   {status} {key.replace('_', ' ').title()}")


# Global configuration instance
config = Config()