"""
Model Registry Management
Handles model versioning, promotion, and lifecycle management
"""

import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Production-grade model registry for managing ML model lifecycle

    Features:
    - Model versioning
    - Stage management (Development, Staging, Production)
    - Automated best model promotion
    - Model comparison and selection
    """

    def __init__(self, registry_uri: Optional[str] = None):
        """
        Initialize Model Registry

        Args:
            registry_uri: MLflow tracking URI (default: local mlruns/)
        """
        if registry_uri:
            mlflow.set_tracking_uri(registry_uri)

        self.client = MlflowClient()
        logger.info(f"✅ Model Registry initialized")

    def register_model(
            self,
            model_uri: str,
            model_name: str,
            tags: Optional[Dict[str, Any]] = None,
            description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Register a new model version

        Args:
            model_uri: URI to the model (e.g., runs:/<run_id>/model)
            model_name: Name for the model
            tags: Dictionary of metadata tags
            description: Model description

        Returns:
            Dictionary with model version info
        """
        try:
            # Register model
            model_version = mlflow.register_model(model_uri, model_name)

            logger.info(f"✅ Registered {model_name} version {model_version.version}")

            # Add tags if provided
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=model_version.version,
                        key=key,
                        value=str(value)
                    )

            # Add description
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description
                )

            # Add registration timestamp
            self.client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="registered_at",
                value=datetime.now().isoformat()
            )

            return {
                "name": model_name,
                "version": model_version.version,
                "run_id": model_version.run_id,
                "status": "registered"
            }

        except Exception as e:
            logger.error(f"❌ Error registering model: {e}")
            raise

    def promote_model(
            self,
            model_name: str,
            version: int,
            stage: str = "Production",
            archive_existing: bool = True
    ) -> Dict[str, str]:
        """
        Promote a model version to a specific stage

        Args:
            model_name: Name of the model
            version: Version number to promote
            stage: Target stage (None, Staging, Production, Archived)
            archive_existing: Whether to archive existing models in target stage

        Returns:
            Dictionary with promotion status
        """
        try:
            # Archive existing models in target stage if requested
            if archive_existing and stage in ["Staging", "Production"]:
                existing_models = self.client.get_latest_versions(
                    name=model_name,
                    stages=[stage]
                )

                for model in existing_models:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=model.version,
                        stage="Archived"
                    )
                    logger.info(f"📦 Archived {model_name} v{model.version}")

            # Promote new version
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )

            # Add promotion timestamp tag
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key=f"promoted_to_{stage.lower()}_at",
                value=datetime.now().isoformat()
            )

            logger.info(f"🚀 Promoted {model_name} v{version} to {stage}")

            return {
                "model_name": model_name,
                "version": version,
                "stage": stage,
                "status": "promoted"
            }

        except Exception as e:
            logger.error(f"❌ Error promoting model: {e}")
            raise

    def get_best_model(
            self,
            model_name: str,
            metric: str = "val_accuracy",
            minimize: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best model version based on a metric

        Args:
            model_name: Name of the model
            metric: Metric to compare (e.g., 'val_accuracy', 'val_loss')
            minimize: If True, lower is better (for loss). If False, higher is better.

        Returns:
            Dictionary with best model info or None
        """
        try:
            # Get all versions
            versions = self.client.search_model_versions(f"name='{model_name}'")

            if not versions:
                logger.warning(f"⚠️  No versions found for {model_name}")
                return None

            best_model = None
            best_metric_value = float('inf') if minimize else float('-inf')

            for version in versions:
                # Get run metrics
                run = self.client.get_run(version.run_id)
                metric_value = run.data.metrics.get(metric)

                if metric_value is None:
                    continue

                # Compare
                is_better = (
                        (minimize and metric_value < best_metric_value) or
                        (not minimize and metric_value > best_metric_value)
                )

                if is_better:
                    best_metric_value = metric_value
                    best_model = {
                        "name": model_name,
                        "version": version.version,
                        "run_id": version.run_id,
                        "stage": version.current_stage,
                        "metric": metric,
                        "metric_value": metric_value,
                        "tags": version.tags
                    }

            if best_model:
                logger.info(
                    f"🏆 Best model: {model_name} v{best_model['version']} "
                    f"({metric}={best_model['metric_value']:.4f})"
                )

            return best_model

        except Exception as e:
            logger.error(f"❌ Error finding best model: {e}")
            raise

    def auto_promote_best_model(
            self,
            model_name: str,
            metric: str = "val_accuracy",
            threshold: Optional[float] = None,
            stage: str = "Production"
    ) -> Optional[Dict[str, Any]]:
        """
        Automatically promote the best model to a stage

        Args:
            model_name: Name of the model
            metric: Metric to compare
            threshold: Minimum metric value required for promotion
            stage: Target stage for promotion

        Returns:
            Dictionary with promotion info or None
        """
        try:
            # Find best model
            best_model = self.get_best_model(model_name, metric)

            if not best_model:
                logger.warning("⚠️  No model found for auto-promotion")
                return None

            # Check threshold
            if threshold and best_model['metric_value'] < threshold:
                logger.info(
                    f"⚠️  Best model ({best_model['metric_value']:.4f}) "
                    f"below threshold ({threshold:.4f}). Skipping promotion."
                )
                return None

            # Check if already in target stage
            if best_model['stage'] == stage:
                logger.info(f"✅ Model already in {stage}")
                return best_model

            # Promote
            promotion_result = self.promote_model(
                model_name=model_name,
                version=best_model['version'],
                stage=stage
            )

            return {**best_model, **promotion_result}

        except Exception as e:
            logger.error(f"❌ Error in auto-promotion: {e}")
            raise

    def list_models(self, stages: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List all registered models

        Args:
            stages: Filter by stages (e.g., ['Production', 'Staging'])

        Returns:
            List of model dictionaries
        """
        try:
            registered_models = self.client.search_registered_models()

            models_info = []
            for model in registered_models:
                versions = self.client.search_model_versions(f"name='{model.name}'")

                for version in versions:
                    # Filter by stage if specified
                    if stages and version.current_stage not in stages:
                        continue

                    models_info.append({
                        "name": model.name,
                        "version": version.version,
                        "stage": version.current_stage,
                        "run_id": version.run_id,
                        "tags": version.tags,
                        "description": version.description
                    })

            return models_info

        except Exception as e:
            logger.error(f"❌ Error listing models: {e}")
            raise

    def get_production_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the current production model

        Args:
            model_name: Name of the model

        Returns:
            Production model info or None
        """
        try:
            production_models = self.client.get_latest_versions(
                name=model_name,
                stages=["Production"]
            )

            if not production_models:
                logger.warning(f"⚠️  No production model found for {model_name}")
                return None

            model = production_models[0]
            return {
                "name": model_name,
                "version": model.version,
                "run_id": model.run_id,
                "stage": model.current_stage,
                "tags": model.tags
            }

        except Exception as e:
            logger.error(f"❌ Error getting production model: {e}")
            raise

    def compare_versions(
            self,
            model_name: str,
            versions: List[int],
            metrics: List[str]
    ) -> Dict[str, Dict[int, float]]:
        """
        Compare specific model versions across metrics

        Args:
            model_name: Name of the model
            versions: List of version numbers to compare
            metrics: List of metric names to compare

        Returns:
            Dictionary mapping metrics to version comparisons
        """
        try:
            comparison = {metric: {} for metric in metrics}

            for version in versions:
                # Get model version
                model_version = self.client.get_model_version(model_name, version)

                # Get run metrics
                run = self.client.get_run(model_version.run_id)

                for metric in metrics:
                    metric_value = run.data.metrics.get(metric)
                    comparison[metric][version] = metric_value

            return comparison

        except Exception as e:
            logger.error(f"❌ Error comparing versions: {e}")
            raise