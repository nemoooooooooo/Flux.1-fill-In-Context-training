import os
import sys
import yaml
import json
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import hashlib
import time
import socket

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def find_free_port(start_port=7860, max_tries=100):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_tries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port in range {start_port}-{start_port+max_tries}")


@dataclass
class DifficultyLevel:
    """Represents a single difficulty level configuration"""
    name: str
    difficulty_blur_sigma: float
    difficulty_gray_alpha: float
    difficulty_brightness: float = 50.0
    level_index: int = 0
    is_sublevel: bool = False
    parent_level: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)
    
    def get_id(self) -> str:
        """Generate unique ID for this level"""
        return f"{self.level_index:03d}_{self.name}"


@dataclass
class ValidationResult:
    """Results from validation comparison"""
    current_level: str
    baseline_level: str
    better_count: int
    neutral_count: int
    worse_count: int
    total_count: int
    promotion_score: float  # (better + neutral) / total
    should_advance: bool
    validation_metadata: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CurriculumState:
    """Persistent state for curriculum training"""
    current_level_idx: int
    current_level: Dict  # DifficultyLevel as dict
    baseline_checkpoint: Optional[str]
    baseline_level: Optional[Dict]  # DifficultyLevel as dict
    current_checkpoint: Optional[str]
    retry_count: int
    training_steps_total: int
    training_history: List[Dict]
    levels: List[Dict]  # List of DifficultyLevel as dicts
    last_action: str  # 'training', 'validation', 'completed'
    validation_pending: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CurriculumManager:
    """Main curriculum learning orchestrator with resume capability"""
    
    def __init__(self, 
                 base_config_path: str = "curriculum/configs/base.yaml",
                 levels_config_path: str = "curriculum/configs/levels.yaml",
                 cache_root: str = "levels",
                 output_root: str = None,
                 state_file: str = "curriculum_state.json"):
        
        self.base_config_path = Path(base_config_path)
        self.levels_config_path = Path(levels_config_path)
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self.base_config = self._load_yaml(self.base_config_path)
        self.levels_config = self._load_yaml(self.levels_config_path)
        
        # Output directory management
        if output_root:
            self.output_root = Path(output_root)
        else:
            self.output_root = Path("curriculum_outputs")
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.output_root / state_file
        
        # Validation thresholds
        self.promote_threshold = self.levels_config.get("promote_threshold", 0.55)
        self.repeat_backoff = self.levels_config.get("repeat_backoff", 0.75)
        self.fail_halve_factor = self.levels_config.get("fail_halve_factor", 0.5)
        self.val_every_steps = self.levels_config.get("val_every_steps", 500)
        self.max_validation_samples = self.levels_config.get("max_validation_samples", None)
        self.max_retries = self.levels_config.get("max_retries", 3)
        
        # Initialize or restore state
        self._restore_or_init_state()
        
        logger.info(f"Initialized CurriculumManager with {len(self.levels)} levels")
        if self.state_file.exists():
            logger.info(f"Restored from checkpoint. Current level: {self.current_level_idx}/{len(self.levels)}")
    
    def _load_yaml(self, path: Path) -> Dict:
        """Load YAML configuration"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _save_state(self):
        """Save current training state"""
        state = CurriculumState(
            current_level_idx=self.current_level_idx,
            current_level=self.get_current_level().to_dict() if self.current_level_idx < len(self.levels) else None,
            baseline_checkpoint=str(self.baseline_checkpoint) if self.baseline_checkpoint else None,
            baseline_level=self.baseline_level.to_dict() if self.baseline_level else None,
            current_checkpoint=str(self.current_checkpoint) if self.current_checkpoint else None,
            retry_count=self.retry_count,
            training_steps_total=self.training_steps_total,
            training_history=self.training_history,
            levels=[l.to_dict() for l in self.levels],
            last_action=self.last_action,
            validation_pending=self.validation_pending
        )
        
        with open(self.state_file, 'w') as f:
            json.dump(asdict(state), f, indent=2)
        logger.info(f"State saved to {self.state_file}")
    
    def _restore_or_init_state(self):
        """Restore state from file or initialize new state"""
        if self.state_file.exists():
            logger.info(f"Restoring state from {self.state_file}")
            with open(self.state_file, 'r') as f:
                state_dict = json.load(f)
            
            # Restore state with defaults for missing fields
            self.current_level_idx = state_dict.get('current_level_idx', 0)
            self.baseline_checkpoint = Path(state_dict['baseline_checkpoint']) if state_dict.get('baseline_checkpoint') else None
            self.current_checkpoint = Path(state_dict['current_checkpoint']) if state_dict.get('current_checkpoint') else None
            self.retry_count = state_dict.get('retry_count', 0)
            self.training_steps_total = state_dict.get('training_steps_total', 0)
            self.training_history = state_dict.get('training_history', [])
            self.last_action = state_dict.get('last_action', 'unknown')
            self.validation_pending = state_dict.get('validation_pending', False)
            
            # Restore levels
            self.levels = []
            for level_dict in state_dict['levels']:
                level = DifficultyLevel(
                    name=level_dict['name'],
                    difficulty_blur_sigma=level_dict['difficulty_blur_sigma'],
                    difficulty_gray_alpha=level_dict['difficulty_gray_alpha'],
                    difficulty_brightness=level_dict.get('difficulty_brightness', 50.0),
                    level_index=level_dict['level_index'],
                    is_sublevel=level_dict.get('is_sublevel', False),
                    parent_level=level_dict.get('parent_level', None)
                )
                self.levels.append(level)
            
            # Restore baseline level
            if state_dict['baseline_level']:
                bl = state_dict['baseline_level']
                self.baseline_level = DifficultyLevel(
                    name=bl['name'],
                    difficulty_blur_sigma=bl['difficulty_blur_sigma'],
                    difficulty_gray_alpha=bl['difficulty_gray_alpha'],
                    difficulty_brightness=bl.get('difficulty_brightness', 50.0),
                    level_index=bl['level_index'],
                    is_sublevel=bl.get('is_sublevel', False),
                    parent_level=bl.get('parent_level', None)
                )
            else:
                self.baseline_level = None
        else:
            # Initialize new state
            self.levels = self._parse_levels()
            self.current_level_idx = 0
            self.retry_count = 0
            self.training_history = []
            self.training_steps_total = 0
            self.current_checkpoint = None
            self.baseline_checkpoint = None
            self.baseline_level = None
            self.last_action = "initialized"
            self.validation_pending = False
    
    def _parse_levels(self) -> List[DifficultyLevel]:
        """Parse difficulty levels from config"""
        levels = []
        for idx, level_data in enumerate(self.levels_config.get("levels", [])):
            level = DifficultyLevel(
                name=level_data["name"],
                difficulty_blur_sigma=level_data["difficulty_blur_sigma"],
                difficulty_gray_alpha=level_data["difficulty_gray_alpha"],
                difficulty_brightness=level_data.get("difficulty_brightness", 50.0),
                level_index=idx
            )
            levels.append(level)
        return levels
    
    def get_current_level(self) -> DifficultyLevel:
        """Get current difficulty level"""
        return self.levels[self.current_level_idx]
    
    def get_level_output_dir(self, level: DifficultyLevel) -> Path:
        """Get output directory for a specific level"""
        return self.cache_root / level.get_id()
    
    def prepare_level_dataset(self, level: DifficultyLevel) -> Path:
        """Prepare transformed dataset for current level"""
        logger.info(f"Preparing dataset for level: {level.name}")
        
        # Check if dataset already exists
        dataset_dirs = list(self.cache_root.glob(f"{level.name}_*"))
        if dataset_dirs:
            logger.info(f"Using existing dataset at: {dataset_dirs[0]}")
            return dataset_dirs[0]
        
        # Create temporary levels config for this specific level
        temp_config = {
            "levels": [{
                "name": level.name,
                "difficulty_blur_sigma": level.difficulty_blur_sigma,
                "difficulty_gray_alpha": level.difficulty_gray_alpha,
                "difficulty_brightness": level.difficulty_brightness
            }]
        }
        
        # Write temporary config
        temp_config_path = self.cache_root / f"temp_level_{level.name}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(temp_config, f)
        
        try:
            # Run transform_dataset.py with real-time output
            cmd = [
                sys.executable,
                "curriculum/transform_dataset.py",
                "--levels_yaml", str(temp_config_path),
                "--base_yaml", str(self.base_config_path),
                "--cache_root", str(self.cache_root),
                "--only_level", level.name
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output to show progress
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"  TRANSFORM: {line.rstrip()}")
            
            return_code = process.wait()
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
            
            # Find the created dataset directory
            dataset_dirs = list(self.cache_root.glob(f"{level.name}_*"))
            if dataset_dirs:
                dataset_path = dataset_dirs[0]
                logger.info(f"Dataset ready at: {dataset_path}")
                return dataset_path
            else:
                raise RuntimeError(f"Dataset directory not found for level {level.name}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to prepare dataset")
            raise
        finally:
            if temp_config_path.exists():
                temp_config_path.unlink()
    
    def run_training_step(self, 
                         level: DifficultyLevel,
                         dataset_path: Path,
                         resume_from: Optional[Path] = None,
                         step_count: int = None) -> Path:
        """Execute training for one validation interval"""
        
        checkpoint_dir = self.get_level_output_dir(level)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        step_count = step_count or self.val_every_steps
        
        logger.info(f"Starting training step for level {level.name}")
        logger.info(f"Training for {step_count} steps")
        logger.info(f"Output: {checkpoint_dir}")
        
        # If resuming, we need to copy the checkpoint to the new directory first
        if resume_from and resume_from.exists():
            logger.info(f"Preparing to resume from: {resume_from}")
            
            # Create checkpoint-0 in the new directory for resuming
            resume_checkpoint_dest = checkpoint_dir / "checkpoint-0"
            
            if not resume_checkpoint_dest.exists():
                logger.info(f"Copying checkpoint from {resume_from} to {resume_checkpoint_dest}")
                shutil.copytree(resume_from, resume_checkpoint_dest)
                
                # For base training mode, also copy the transformer directory if it exists
                if self.base_config.get("train_mode", "lora") == "base":
                    src_transformer = resume_from.parent / "transformer"
                    if src_transformer.exists():
                        dst_transformer = checkpoint_dir / "transformer"
                        if not dst_transformer.exists():
                            logger.info(f"Copying transformer weights")
                            shutil.copytree(src_transformer, dst_transformer)
            
            # Update resume path to the copied checkpoint
            resume_from = resume_checkpoint_dest
        
        # Build training command
        cmd = ["accelerate", "launch", "train.py"]
        
        # Add base config parameters
        for key, value in self.base_config.items():
            if key in ["output_dir", "dataset_name", "gaussian_blur", "random_grayscale", 
                      "validation_steps", "checkpointing_steps", "max_train_steps"]:
                continue
            
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        # Add level-specific overrides
        cmd.extend([
            "--output_dir", str(checkpoint_dir),
            "--dataset_name", str(dataset_path),
            "--gaussian_blur", "0.0",  # Already applied
            "--random_grayscale", "0.0",  # Already applied
            "--validation_steps", "999999",  # Skip validation during training
            "--checkpointing_steps", str(step_count),
            "--max_train_steps", str(step_count),  # Train for one interval
        ])
        
        if self.max_validation_samples:
            cmd.extend(["--max_validation_samples", str(self.max_validation_samples)])
        
        if resume_from:
            cmd.extend(["--resume_from_checkpoint", str(resume_from)])
        
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info("Starting training (this may take a few minutes)...")
        
        try:
            # Run training with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.rstrip())  # Print to console for real-time monitoring
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
            
            # Find the created checkpoint
            checkpoint = checkpoint_dir / f"checkpoint-{step_count}"
            if not checkpoint.exists():
                # Sometimes might be named differently
                checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
                if checkpoints:
                    checkpoint = checkpoints[-1]
                else:
                    raise RuntimeError("No checkpoint created")
            
            # Clean up the temporary checkpoint-0 if it exists
            if resume_from and resume_from.name == "checkpoint-0":
                logger.info(f"Cleaning up temporary checkpoint: {resume_from}")
                shutil.rmtree(resume_from)
            
            logger.info(f"Training completed. Checkpoint: {checkpoint}")
            return checkpoint
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed with return code: {e.returncode}")
            raise
    
    def generate_validation_images(self, checkpoint: Path, level: DifficultyLevel) -> Path:
        """Generate validation images for a checkpoint"""
        val_dir = checkpoint.parent / "val"
        
        # Check if validation images already exist
        if val_dir.exists() and any(val_dir.glob("*.png")):
            logger.info(f"Using existing validation images at {val_dir}")
            return val_dir
        
        logger.info(f"Need to generate validation images for {level.name}")
        logger.info(f"Please run validation generation manually for checkpoint: {checkpoint}")
        logger.info(f"Expected output directory: {val_dir}")
        
        # For now, we'll assume validation images exist from training
        # In practice, you'd call your validation script here
        return val_dir
    
    def run_human_validation(self, 
                            val_dir_a: Path,
                            val_dir_b: Path,
                            level_a: DifficultyLevel,
                            level_b: DifficultyLevel) -> ValidationResult:
        """Run human validation using Gradio UI"""
        
        logger.info(f"Launching human validation UI...")
        logger.info(f"Comparing {level_a.name} (baseline) vs {level_b.name} (current)")
        
        # Find a free port
        port = find_free_port(7860)
        logger.info(f"Using port {port} for UI")
        
        # Set environment variable for port
        env = os.environ.copy()
        env['GRADIO_SERVER_PORT'] = str(port)
        
        # Launch the comparison UI
        cmd = [
            sys.executable,
            "curriculum/ui_compare.py",
            str(val_dir_a),
            str(val_dir_b),
            level_a.name,
            level_b.name,
            "--out", str(self.output_root / "validation_results.json"),
            "--port", str(port)
        ]
        
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            logger.error(f"UI failed: {e}")
            raise
        
        # Read results
        results_file = self.output_root / "validation_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            total = results["total_comparisons"]
            better = results["better_count"]
            neutral = results["neutral_count"]
            worse = results["worse_count"]
            
            promotion_score = (better + neutral) / total if total > 0 else 0
            
            return ValidationResult(
                current_level=level_b.name,
                baseline_level=level_a.name,
                better_count=better,
                neutral_count=neutral,
                worse_count=worse,
                total_count=total,
                promotion_score=promotion_score,
                should_advance=promotion_score >= self.promote_threshold,
                validation_metadata=results
            )
        else:
            logger.error("No validation results found")
            raise RuntimeError("Validation results not saved")
    
    def create_intermediate_level(self, 
                                 baseline_level: DifficultyLevel,
                                 target_level: DifficultyLevel,
                                 retry_num: int) -> DifficultyLevel:
        """Create intermediate difficulty level between baseline and target"""
        
        # Progressive interpolation - gets closer to target with each retry
        # retry 0 → 0.5, retry 1 → 0.3, retry 2 → 0.1
        factor = max(0.5 - (0.2 * retry_num), 0.05)
        
        new_sigma = baseline_level.difficulty_blur_sigma + \
                   (target_level.difficulty_blur_sigma - baseline_level.difficulty_blur_sigma) * factor
        new_alpha = baseline_level.difficulty_gray_alpha + \
                   (target_level.difficulty_gray_alpha - baseline_level.difficulty_gray_alpha) * factor
        new_brightness = baseline_level.difficulty_brightness + \
                        (target_level.difficulty_brightness - baseline_level.difficulty_brightness) * factor
        
        new_level = DifficultyLevel(
            name=f"{baseline_level.name}_to_{target_level.name}_v{retry_num+1}",
            difficulty_blur_sigma=round(new_sigma, 2),
            difficulty_gray_alpha=round(new_alpha, 2),
            difficulty_brightness=round(new_brightness, 1),
            level_index=target_level.level_index,  # Keep target index for ordering
            is_sublevel=True,
            parent_level=baseline_level.name
        )
        
        logger.info(f"Created intermediate level: {new_level.name}")
        logger.info(f"  Blur: {baseline_level.difficulty_blur_sigma} -> {new_level.difficulty_blur_sigma} -> {target_level.difficulty_blur_sigma}")
        logger.info(f"  Gray: {baseline_level.difficulty_gray_alpha} -> {new_level.difficulty_gray_alpha} -> {target_level.difficulty_gray_alpha}")
        logger.info(f"  Brightness: {baseline_level.difficulty_brightness} -> {new_level.difficulty_brightness} -> {target_level.difficulty_brightness}")
        
        return new_level
    
    def resume_from_validation(self):
        """Resume from a pending validation state"""
        if not self.validation_pending:
            logger.info("No pending validation")
            return None
        
        logger.info("Resuming from pending validation...")
        
        current_level = self.get_current_level()
        
        # Generate validation images for current checkpoint
        val_dir_current = self.generate_validation_images(self.current_checkpoint, current_level)
        
        # Generate validation for baseline
        val_dir_baseline = self.generate_validation_images(
            self.baseline_checkpoint, 
            self.baseline_level
        )
        
        # Run validation
        validation_result = self.run_human_validation(
            val_dir_baseline,
            val_dir_current,
            self.baseline_level,
            current_level
        )
        
        # Mark validation as complete
        self.validation_pending = False
        self.last_action = "validation_completed"
        self._save_state()
        
        return validation_result
    
    def run_curriculum(self, start_level: int = None, resume: bool = True):
        """Main curriculum training loop with resume capability"""
        
        if resume and self.state_file.exists():
            logger.info("="*80)
            logger.info("RESUMING Curriculum Training")
            logger.info(f"Last action: {self.last_action}")
            logger.info(f"Current level: {self.current_level_idx}/{len(self.levels)}")
            logger.info(f"Total steps so far: {self.training_steps_total}")
            logger.info("="*80)
            
            # Check if we have pending validation
            if self.validation_pending:
                logger.info("Pending validation detected. Running validation first...")
                validation_result = self.resume_from_validation()
                if validation_result:
                    self._process_validation_result(validation_result)
        else:
            logger.info("="*80)
            logger.info("Starting NEW Curriculum Training")
            logger.info(f"Levels: {[l.name for l in self.levels]}")
            logger.info(f"Promotion threshold: {self.promote_threshold}")
            logger.info("="*80)
            
            if start_level is not None:
                self.current_level_idx = start_level
        
        while self.current_level_idx < len(self.levels):
            current_level = self.get_current_level()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"LEVEL {current_level.level_index}: {current_level.name}")
            if current_level.is_sublevel:
                logger.info(f"  (Intermediate level - retry {self.retry_count})")
            logger.info(f"{'='*60}")
            
            # Prepare dataset
            dataset_path = self.prepare_level_dataset(current_level)
            
            # Determine checkpoint to resume from
            resume_checkpoint = self.baseline_checkpoint if self.baseline_checkpoint else None
            
            # Train for one validation interval
            self.last_action = "training"
            self._save_state()
            
            self.current_checkpoint = self.run_training_step(
                current_level,
                dataset_path,
                resume_from=resume_checkpoint,
                step_count=self.val_every_steps
            )
            
            self.training_steps_total += self.val_every_steps
            
            # Mark that validation is pending
            self.validation_pending = True
            self.last_action = "training_completed"
            self._save_state()
            
            # Generate validation images for current checkpoint
            val_dir_current = self.generate_validation_images(self.current_checkpoint, current_level)
            
            # First level becomes baseline
            if self.baseline_checkpoint is None:
                self.baseline_checkpoint = self.current_checkpoint
                self.baseline_level = current_level
                logger.info(f"✅ Baseline established: {current_level.name}")
                self.current_level_idx += 1
                self.retry_count = 0
                self.validation_pending = False
                self.last_action = "baseline_established"
                self._save_state()
                continue
            
            # Generate validation for baseline (if not already done)
            val_dir_baseline = self.generate_validation_images(
                self.baseline_checkpoint, 
                self.baseline_level
            )
            
            # Run human validation
            logger.info("\n" + "="*40)
            logger.info("VALIDATION CHECK")
            logger.info(f"Baseline: {self.baseline_level.name}")
            logger.info(f"Current: {current_level.name}")
            logger.info("="*40)
            
            validation_result = self.run_human_validation(
                val_dir_baseline,
                val_dir_current,
                self.baseline_level,
                current_level
            )
            
            self.validation_pending = False
            self._process_validation_result(validation_result)
        
        logger.info("\n" + "="*80)
        logger.info("CURRICULUM TRAINING COMPLETE!")
        logger.info(f"Final checkpoint: {self.baseline_checkpoint}")
        logger.info(f"Final level achieved: {self.baseline_level.name}")
        logger.info(f"Total training steps: {self.training_steps_total}")
        logger.info("="*80)
        
        # Save final model
        if self.baseline_checkpoint:
            final_model_dir = self.output_root / "final_model"
            logger.info(f"Saving final model to {final_model_dir}")
            final_model_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.baseline_checkpoint, final_model_dir / "checkpoint-final")
            
            # For base mode, also copy transformer
            if self.base_config.get("train_mode", "lora") == "base":
                src_transformer = self.baseline_checkpoint.parent / "transformer"
                if src_transformer.exists():
                    shutil.copytree(src_transformer, final_model_dir / "transformer")
        
        self.last_action = "completed"
        self._save_state()
        
        return self.baseline_checkpoint
    
    def _process_validation_result(self, validation_result: ValidationResult):
        """Process validation result and update state accordingly"""
        current_level = self.get_current_level()
        
        # Log result
        self.training_history.append({
            "timestamp": validation_result.timestamp,
            "baseline_level": self.baseline_level.to_dict(),
            "current_level": current_level.to_dict(),
            "result": asdict(validation_result),
            "total_steps": self.training_steps_total,
            "retry_count": self.retry_count
        })
        
        # Save progress
        with open(self.output_root / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Decision logic
        logger.info(f"\nValidation Score: {validation_result.promotion_score:.2%}")
        logger.info(f"Threshold: {self.promote_threshold:.2%}")
        
        if validation_result.promotion_score >= self.promote_threshold:
            logger.info(f"✅ PROMOTED! Model improved on {current_level.name}")
            
            # Update baseline to current
            self.baseline_checkpoint = self.current_checkpoint
            self.baseline_level = current_level
            self.current_level_idx += 1
            self.retry_count = 0
            self.last_action = "promoted"
            
        elif self.retry_count < self.max_retries:
            logger.info(f"❌ Not ready for {current_level.name} (score: {validation_result.promotion_score:.2%})")
            logger.info(f"Creating intermediate level (retry {self.retry_count + 1}/{self.max_retries})")
            
            # Create intermediate level between baseline and current target
            target_level = (current_level if not current_level.is_sublevel else
                            next(l for l in self.levels if (not l.is_sublevel and l.level_index == current_level.level_index)))
            intermediate = self.create_intermediate_level(self.baseline_level, target_level, self.retry_count)

            # Insert at the current position so the original target shifts right and stays next
            self.levels.insert(self.current_level_idx, intermediate)
            logger.info(f"Intermediate level created: {self.levels}")
            self.last_action = "retry_with_intermediate"
            self.retry_count += 1
            
        else:
            logger.info(f"❌ Max retries ({self.max_retries}) reached.")
            logger.info(f"Staying at {self.baseline_level.name} and moving to next level")
            
            # Skip to next original level
            while self.current_level_idx < len(self.levels) - 1:
                self.current_level_idx += 1
                if not self.levels[self.current_level_idx].is_sublevel:
                    break
            else:
                self.current_level_idx += 1
            
            self.retry_count = 0
            self.last_action = "skipped_to_next"
        
        self._save_state()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Curriculum Learning Manager")
    parser.add_argument("--start-level", type=int, default=None, help="Start from specific level")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore saved state")
    parser.add_argument("--resume-validation", action="store_true", help="Only run pending validation")
    parser.add_argument("--base-config", type=str, default="curriculum/configs/base.yaml")
    parser.add_argument("--levels-config", type=str, default="curriculum/configs/levels.yaml")
    parser.add_argument("--cache-root", type=str, default="levels")
    parser.add_argument("--output-root", type=str, default="curriculum_outputs")
    
    args = parser.parse_args()
    
    manager = CurriculumManager(
        base_config_path=args.base_config,
        levels_config_path=args.levels_config,
        cache_root=args.cache_root,
        output_root=args.output_root
    )
    
    if args.resume_validation:
        # Only run pending validation if it exists
        if manager.validation_pending:
            validation_result = manager.resume_from_validation()
            if validation_result:
                manager._process_validation_result(validation_result)
        else:
            logger.info("No pending validation to resume")
    else:
        # Run full curriculum
        manager.run_curriculum(
            start_level=args.start_level,
            resume=not args.no_resume
        )