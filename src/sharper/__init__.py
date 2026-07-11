"""Sharper's public package interface."""

from sharper.analysis import (
    CategoricalAnalysis,
    CorrelationAnalysis,
    GroupComparison,
    NumericAnalysis,
    OutlierAnalysis,
    TargetAnalysis,
    analyze_categorical_features,
    analyze_numeric_features,
    analyze_target_relationships,
    compare_groups,
    compute_correlations,
    detect_outliers,
)
from sharper.features import (
    FeatureDerivationResult,
    FeatureSuggestion,
    FeatureSuggestionReport,
    derive_features,
    suggest_feature_derivations,
)
from sharper.io import load_csv, load_excel
from sharper.quality import QualityIssue, QualityReport, check_data_quality
from sharper.reporting import ReportArtifact, generate_analysis_report
from sharper.schema import ColumnSchema, SchemaReport, TargetCandidate, infer_schema
from sharper.summary import DataFrameSummary, summarize_dataframe
from sharper.workflow import AnalysisRun, run_analysis

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "load_csv",
    "load_excel",
    "ColumnSchema",
    "TargetCandidate",
    "SchemaReport",
    "infer_schema",
    "DataFrameSummary",
    "summarize_dataframe",
    "QualityIssue",
    "QualityReport",
    "check_data_quality",
    "AnalysisRun",
    "run_analysis",
    "ReportArtifact",
    "generate_analysis_report",
    "NumericAnalysis",
    "CategoricalAnalysis",
    "CorrelationAnalysis",
    "OutlierAnalysis",
    "analyze_numeric_features",
    "analyze_categorical_features",
    "compute_correlations",
    "detect_outliers",
    "GroupComparison",
    "TargetAnalysis",
    "compare_groups",
    "analyze_target_relationships",
    "FeatureSuggestion",
    "FeatureSuggestionReport",
    "FeatureDerivationResult",
    "suggest_feature_derivations",
    "derive_features",
]
