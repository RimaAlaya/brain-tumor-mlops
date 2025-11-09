"""
Enhanced Experiment Tracking
Comprehensive logging and monitoring for ML experiments
"""

import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import psutil
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Enhanced experiment tracking with comprehensive logging

    Features:
    - Parameter and metric logging
    - System metrics (CPU, GPU, memory)
    - Artifacts management
    - Git integration
    - Custom tags and metadata
    """

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """
        Initialize Experiment Tracker

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.experiment_name = experiment_name
        self.client = MlflowClient()

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)
        logger.info(f"✅ Experiment tracker initialized: {experiment_name}")

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run with enhanced logging

        Args:
            run_name: Name for this run
            tags: Dictionary of tags
        """
        self.run = mlflow.start_run(run_name=run_name)

        # Log system info
        self._log_system_info()

        # Log custom tags
        if tags:
            mlflow.set_tags(tags)

        # Log timestamp
        mlflow.set_tag("start_time", datetime.now().isoformat())

        logger.info(f"🏃 Started run: {self.run.info.run_id}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters with validation

        Args:
            params: Dictionary of parameters
        """
        try:
            # MLflow params must be strings, convert if needed
            string_params = {k: str(v) if not isinstance(v, str) else v for k, v in params.items()}
            mlflow.log_params(string_params)
            logger.info(f"📝 Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"❌ Error logging parameters: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics with optional step

        Args:
            metrics: Dictionary of metrics
            step: Training step/epoch number
        """
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.info(f"📊 Logged {len(metrics)} metrics")
        except Exception as e:
            logger.error(f"❌ Error logging metrics: {e}")

    def log_metric_history(self, metric_name: str, values: List[float]):
        """
        Log a series of metric values (e.g., per-epoch)

        Args:
            metric_name: Name of the metric
            values: List of metric values
        """
        try:
            for step, value in enumerate(values):
                mlflow.log_metric(metric_name, value, step=step)
            logger.info(f"📈 Logged {metric_name} history ({len(values)} steps)")
        except Exception as e:
            logger.error(f"❌ Error logging metric history: {e}")

    def log_model(self, model, artifact_path: str = "model", **kwargs):
        """
        Log trained model with metadata

        Args:
            model: Trained model object
            artifact_path: Path within run artifacts
            **kwargs: Additional arguments for model logging
        """
        try:
            mlflow.keras.log_model(model, artifact_path, **kwargs)
            logger.info(f"💾 Model logged to {artifact_path}")
        except Exception as e:
            logger.error(f"❌ Error logging model: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a file as an artifact

        Args:
            local_path: Path to local file
            artifact_path: Path within artifacts directory
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"📎 Artifact logged: {local_path}")
        except Exception as e:
            logger.error(f"❌ Error logging artifact: {e}")

    def log_figure(self, figure, filename: str):
        """
        Log a matplotlib figure

        Args:
            figure: Matplotlib figure object
            filename: Name for the saved figure
        """
        try:
            mlflow.log_figure(figure, filename)
            logger.info(f"📊 Figure logged: {filename}")
        except Exception as e:
            logger.error(f"❌ Error logging figure: {e}")

    def log_dict(self, dictionary: Dict, filename: str):
        """
        Log a dictionary as JSON artifact

        Args:
            dictionary: Dictionary to log
            filename: Name for the JSON file
        """
        try:
            mlflow.log_dict(dictionary, filename)
            logger.info(f"📋 Dictionary logged: {filename}")
        except Exception as e:
            logger.error(f"❌ Error logging dictionary: {e}")

    def log_confusion_matrix(self, cm, class_names: List[str], filename: str = "confusion_matrix.png"):
        """
        Log confusion matrix as artifact

        Args:
            cm: Confusion matrix array
            class_names: List of class names
            filename: Filename for saved figure
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(10, 8))

            # Normalize
            cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, None]

            # Plot
            sns.heatmap(
                cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax
            )

            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix (Normalized)")

            plt.tight_layout()
            self.log_figure(fig, filename)
            plt.close()

            logger.info(f"📊 Confusion matrix logged")
        except Exception as e:
            logger.error(f"❌ Error logging confusion matrix: {e}")

    def log_training_curves(self, history, filename: str = "training_curves.png"):
        """
        Log training history curves

        Args:
            history: Keras training history object
            filename: Filename for saved figure
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Accuracy
            axes[0].plot(history.history["accuracy"], label="Train", linewidth=2)
            axes[0].plot(history.history["val_accuracy"], label="Validation", linewidth=2)
            axes[0].set_title("Model Accuracy", fontsize=14)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Accuracy")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Loss
            axes[1].plot(history.history["loss"], label="Train", linewidth=2)
            axes[1].plot(history.history["val_loss"], label="Validation", linewidth=2)
            axes[1].set_title("Model Loss", fontsize=14)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            self.log_figure(fig, filename)
            plt.close()

            logger.info(f"📈 Training curves logged")
        except Exception as e:
            logger.error(f"❌ Error logging training curves: {e}")

    def _log_system_info(self):
        """Log system and environment information"""
        try:
            # Platform info
            mlflow.set_tag("os", platform.system())
            mlflow.set_tag("os_version", platform.version())
            mlflow.set_tag("python_version", platform.python_version())

            # Hardware info
            mlflow.set_tag("cpu_count", psutil.cpu_count())
            mlflow.set_tag("memory_gb", round(psutil.virtual_memory().total / (1024**3), 2))

            # Try to get GPU info
            try:
                import tensorflow as tf

                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    mlflow.set_tag("gpu_available", "yes")
                    mlflow.set_tag("gpu_count", len(gpus))
                else:
                    mlflow.set_tag("gpu_available", "no")
            except:
                mlflow.set_tag("gpu_available", "unknown")

            logger.info("💻 System info logged")
        except Exception as e:
            logger.error(f"⚠️  Could not log all system info: {e}")

    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """
        Log dataset metadata

        Args:
            dataset_info: Dictionary with dataset information
        """
        try:
            for key, value in dataset_info.items():
                mlflow.set_tag(f"dataset_{key}", str(value))

            logger.info("📊 Dataset info logged")
        except Exception as e:
            logger.error(f"❌ Error logging dataset info: {e}")

    def log_code_version(self):
        """Log git commit hash if available"""
        try:
            import subprocess

            commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()

            mlflow.set_tag("git_commit", commit_hash)
            logger.info(f"📌 Git commit logged: {commit_hash[:7]}")
        except:
            logger.warning("⚠️  Could not log git commit (not a git repo)")

    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        try:
            # Log end time
            mlflow.set_tag("end_time", datetime.now().isoformat())

            # Calculate duration
            start_time = mlflow.get_run(self.run.info.run_id).data.tags.get("start_time")
            if start_time:
                start = datetime.fromisoformat(start_time)
                duration = (datetime.now() - start).total_seconds()
                mlflow.log_metric("duration_seconds", duration)

            mlflow.end_run(status=status)
            logger.info(f"✅ Run ended: {self.run.info.run_id}")
        except Exception as e:
            logger.error(f"❌ Error ending run: {e}")

    def log_classification_report(self, y_true, y_pred, class_names: List[str], filename: str = "classification_report.json"):
        """
        Log classification report

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            filename: Filename for report
        """
        try:
            from sklearn.metrics import classification_report

            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

            # Log as artifact
            self.log_dict(report, filename)

            # Log key metrics
            mlflow.log_metric("macro_f1", report["macro avg"]["f1-score"])
            mlflow.log_metric("weighted_f1", report["weighted avg"]["f1-score"])

            logger.info("📋 Classification report logged")
        except Exception as e:
            logger.error(f"❌ Error logging classification report: {e}")


# Context manager for automatic run management
class ManagedRun:
    """Context manager for MLflow runs"""

    def __init__(self, tracker: ExperimentTracker, run_name: Optional[str] = None, **tags):
        self.tracker = tracker
        self.run_name = run_name
        self.tags = tags

    def __enter__(self):
        self.tracker.start_run(self.run_name, self.tags)
        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.tracker.end_run(status="FAILED")
        else:
            self.tracker.end_run(status="FINISHED")
