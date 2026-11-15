"""DataGate: pre-ingestion checkpoint for training data."""

from datagate.orchestrator import run_analysis
from datagate.types import ProjectSpec, Report, ScoredEvidence

__version__ = "0.1.0"
__all__ = ["run_analysis", "ProjectSpec", "Report", "ScoredEvidence"]
