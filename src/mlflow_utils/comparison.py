"""
Experiment Comparison Utilities
Compare and analyze multiple ML experiments
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from typing import List, Optional, Dict, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentComparison:
    """
    Compare and analyze ML experiments

    Features:
    - Compare runs across metrics
    - Generate comparison reports
    - Visualize experiment results
    - Find optimal hyperparameters
    """

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """
        Initialize Experiment Comparison

        Args:
            experiment_name: Name of the experiment to analyze
            tracking_uri: MLflow tracking URI
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.experiment_name = experiment_name
        self.client = MlflowClient()

        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            self.experiment_id = experiment.experiment_id
        else:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        logger.info(f"✅ Experiment comparison initialized: {experiment_name}")

    def get_all_runs(self, filter_string: Optional[str] = None) -> pd.DataFrame:
        """
        Get all runs from the experiment as DataFrame

        Args:
            filter_string: MLflow search filter (e.g., "metrics.accuracy > 0.9")

        Returns:
            DataFrame with run information
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                order_by=["start_time DESC"]
            )

            logger.info(f"📊 Retrieved {len(runs)} runs")
            return runs

        except Exception as e:
            logger.error(f"❌ Error retrieving runs: {e}")
            raise

    def compare_runs(
            self,
            run_ids: List[str],
            metrics: Optional[List[str]] = None,
            params: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare specific runs

        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare
            params: List of parameters to compare

        Returns:
            Comparison DataFrame
        """
        try:
            comparison_data = []

            for run_id in run_ids:
                run = self.client.get_run(run_id)

                row = {
                    "run_id": run_id[:8],  # Shortened ID
                    "run_name": run.data.tags.get("mlflow.runName", "N/A"),
                    "status": run.info.status,
                    "start_time": run.info.start_time
                }

                # Add metrics
                if metrics:
                    for metric in metrics:
                        row[f"metric_{metric}"] = run.data.metrics.get(metric)
                else:
                    # Add all metrics
                    for metric, value in run.data.metrics.items():
                        row[f"metric_{metric}"] = value

                # Add parameters
                if params:
                    for param in params:
                        row[f"param_{param}"] = run.data.params.get(param)
                else:
                    # Add all params
                    for param, value in run.data.params.items():
                        row[f"param_{param}"] = value

                comparison_data.append(row)

            df = pd.DataFrame(comparison_data)
            logger.info(f"📊 Compared {len(run_ids)} runs")

            return df

        except Exception as e:
            logger.error(f"❌ Error comparing runs: {e}")
            raise

    def get_top_runs(
            self,
            metric: str,
            n: int = 5,
            ascending: bool = False
    ) -> pd.DataFrame:
        """
        Get top N runs by metric

        Args:
            metric: Metric to sort by
            n: Number of top runs to return
            ascending: If True, lower is better

        Returns:
            DataFrame with top runs
        """
        try:
            runs = self.get_all_runs()

            metric_col = f"metrics.{metric}"
            if metric_col not in runs.columns:
                logger.error(f"❌ Metric '{metric}' not found")
                return pd.DataFrame()

            # Sort and get top N
            top_runs = runs.sort_values(
                by=metric_col,
                ascending=ascending
            ).head(n)

            logger.info(f"🏆 Top {n} runs by {metric}")

            return top_runs

        except Exception as e:
            logger.error(f"❌ Error getting top runs: {e}")
            raise

    def plot_metric_comparison(
            self,
            metric: str,
            n_runs: int = 10,
            save_path: Optional[str] = None
    ):
        """
        Plot metric comparison across runs

        Args:
            metric: Metric to plot
            n_runs: Number of runs to include
            save_path: Path to save figure
        """
        try:
            runs = self.get_all_runs()

        metric_col = f"metrics.{metric}"
        if metric_col not in runs.columns:
            logger.error(f"❌ Metric '{metric}' not found")
            return

        # Get top N by this metric
        top_runs = runs.nlargest(n_runs, metric_col)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        run_names = [
            run.get("tags.mlflow.runName", f"Run {i}")
            for i, run in top_runs.iterrows()
        ]

        ax.barh(range(len(top_runs)), top_runs[metric_col])
        ax.set_yticks(range(len(top_runs)))
        ax.set_yticklabels(run_names)
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_title(f'Top {n_runs} Runs by {metric}')
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"💾 Plot saved: {save_path}")

        plt.show()

    except Exception as e:
    logger.error(f"❌ Error plotting metric comparison: {e}")


def plot_metric_over_time(
        self,
        metric: str,
        save_path: Optional[str] = None
):
    """
    Plot how a metric changes over time (across runs)

    Args:
        metric: Metric to plot
        save_path: Path to save figure
    """
    try:
        runs = self.get_all_runs()

        metric_col = f"metrics.{metric}"
        if metric_col not in runs.columns:
            logger.error(f"❌ Metric '{metric}' not found")
            return

        # Sort by start time
        runs = runs.sort_values('start_time')

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            runs['start_time'],
            runs[metric_col],
            marker='o',
            linewidth=2,
            markersize=8
        )

        ax.set_xlabel('Time')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric} Over Time')
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"💾 Plot saved: {save_path}")

        plt.show()

    except Exception as e:
        logger.error(f"❌ Error plotting metric over time: {e}")


def analyze_hyperparameters(
        self,
        target_metric: str,
        params: List[str]
) -> pd.DataFrame:
    """
    Analyze correlation between hyperparameters and metric

    Args:
        target_metric: Metric to optimize
        params: List of parameters to analyze

    Returns:
        DataFrame with correlation analysis
    """
    try:
        runs = self.get_all_runs()

        # Prepare data
        metric_col = f"metrics.{target_metric}"
        param_cols = [f"params.{p}" for p in params]

        # Select relevant columns
        available_cols = [metric_col] + [c for c in param_cols if c in runs.columns]
        analysis_df = runs[available_cols].copy()

        # Convert params to numeric where possible
        for col in param_cols:
            if col in analysis_df.columns:
                analysis_df[col] = pd.to_numeric(analysis_df[col], errors='ignore')

        # Calculate correlations
        correlations = analysis_df.corr()[metric_col].drop(metric_col)

        logger.info(f"📊 Hyperparameter analysis complete")

        return correlations.to_frame(name='correlation').sort_values('correlation', ascending=False)

    except Exception as e:
        logger.error(f"❌ Error analyzing hyperparameters: {e}")
        raise


def generate_comparison_report(self, output_path: str = "comparison_report.html"):
    """
    Generate HTML comparison report

    Args:
        output_path: Path for output HTML file
    """
    try:
        runs = self.get_all_runs()

        # Create HTML report
        html = f"""
                    <html>
                    <head>
                        <title>MLflow Experiment Comparison Report</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            h1 {{ color: #2c3e50; }}
                            table {{ border-collapse: collapse; width: 100%; }}
                            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                            th {{ background-color: #3498db; color: white; }}
                            tr:nth-child(even) {{ background-color: #f2f2f2; }}
                        </style>
                    </head>
                    <body>
                        <h1>🧠 Experiment Comparison Report</h1>
                        <h2>Experiment: {self.experiment_name}</h2>
                        <p><strong>Total Runs:</strong> {len(runs)}</p>
                        <p><strong>Generated:</strong> {pd.Timestamp.now()}</p>

                        <h2>📊 Run Summary</h2>
                        {runs.to_html(index=False, classes='table')}
                    </body>
                    </html>
                    """

        with open(output_path, 'w') as f:
            f.write(html)

        logger.info(f"📄 Report generated: {output_path}")

    except Exception as e:
        logger.error(f"❌ Error generating report: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize comparison
    comparison = ExperimentComparison("brain-tumor-efficientnet-clean")

    # Get top 5 runs
    top_runs = comparison.get_top_runs("val_accuracy", n=5)
    print("\n🏆 Top 5 Runs:")
    print(top_runs[['tags.mlflow.runName', 'metrics.val_accuracy']])

    # Plot comparison
    comparison.plot_metric_comparison("val_accuracy", n_runs=10)

    # Generate report
    comparison.generate_comparison_report()
