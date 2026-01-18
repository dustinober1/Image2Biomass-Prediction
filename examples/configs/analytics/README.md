# Advanced Analytics Features

This guide covers the advanced analytics features available in MLflow Tracking, including error analysis, model interpretability, automated insights, and professional report generation.

## Overview

The advanced analytics module provides tools for:

- **Error Analysis**: Analyze prediction errors, identify systematic failure patterns, and visualize residuals
- **Model Interpretability**: Explain model predictions using SHAP values and permutation importance
- **Automated Insights**: Generate statistical comparisons, hyperparameter correlations, and experiment rankings
- **Report Generation**: Create professional HTML/PDF reports combining all analyses

These features help you understand what drives model performance, identify systematic biases, and make data-driven decisions about model improvements.

## Installation

Install the required dependencies:

```bash
# Core analytics dependencies
pip install shap eli5 statsmodels scipy

# Optional: For PDF report generation
pip install weasyprint

# Optional: For interactive plots
pip install plotly

# Verify installation
python -c "import shap, eli5, scipy; print('Analytics dependencies installed successfully')"
```

### Dependencies

- **shap** (>=0.46.0): SHAP values for model interpretability
- **eli5** (>=0.13.0): Permutation importance calculations
- **scipy** (>=1.11.0): Statistical testing (t-test, Mann-Whitney U)
- **statsmodels** (>=0.14.0): Statistical analysis and effect size calculation
- **weasyprint** (>=60.0, optional): PDF report generation
- **jinja2** (>=3.1.0): HTML template rendering

## CLI Usage

### Error Analysis

Analyze prediction errors and identify failure modes:

```bash
# Basic error analysis
exp-analyze-errors <run_id>

# Custom output directory
exp-analyze-errors <run_id> --output-dir results/error_analysis/

# Skip MLflow logging (local only)
exp-analyze-errors <run_id> --no-log-artifacts

# Verbose output
exp-analyze-errors <run_id> --verbose

# Custom predictions artifact path
exp-analyze-errors <run_id> --predictions-path val_predictions.csv
```

**Output:**
- Residual plot (predicted vs actual residuals)
- Error distribution plot
- Prediction vs actual plot
- Failure mode clusters with statistics
- Error statistics table (mean, median, std, percentiles)

**Example:**
```bash
# Analyze errors for a specific run
exp-analyze-errors abc123def456 --verbose

# Output:
# Loading predictions from run abc123def456...
# Generating error analysis plots...
# Identifying failure modes...
#
# ============================================================
# Error Analysis Summary
# ============================================================
# Run ID: abc123def456
#
# Error Statistics:
#   Mean Absolute Error: 12.3456
#   Median Absolute Error: 10.2345
#   Max Absolute Error: 45.6789
#   Std Residual: 8.7654
#   90th Percentile: 22.3456
#   95th Percentile: 28.9012
#   99th Percentile: 38.1234
#
# Failure Modes:
#   Cluster 0:
#     Samples: 45
#     Mean Abs Error: 5.2341
#     Mean Pct Error: 8.45%
#   Cluster 1:
#     Samples: 32
#     Mean Abs Error: 15.6789
#     Mean Pct Error: 22.34%
#   Cluster 2:
#     Samples: 23
#     Mean Abs Error: 32.4567
#     Mean Pct Error: 45.67%
```

### Model Interpretability

Generate SHAP-based model explanations:

```bash
# Basic interpretability analysis
exp-interpret <run_id>

# Different plot types
exp-interpret <run_id> --plot-type bar
exp-interpret <run_id> --plot-type summary
exp-interpret <run_id> --plot-type dependence

# Show more features
exp-interpret <run_id> --max-features 30

# Include permutation importance
exp-interpret <run_id> --compute-permutation

# Custom output directory
exp-interpret <run_id> --output-dir results/interpretability/

# Verbose output
exp-interpret <run_id> --verbose
```

**Output:**
- SHAP summary plot (global feature importance)
- Local explanation plots (individual predictions)
- Permutation importance rankings (if requested)
- Feature importance statistics

**Example:**
```bash
# Generate SHAP analysis
exp-interpret abc123def456 --compute-permutation --verbose

# Output:
# Loading model from run abc123def456...
# Computing SHAP values...
# Generating feature importance plots...
# Computing permutation importance...
#
# ============================================================
# Interpretability Analysis Summary
# ============================================================
# Run ID: abc123def456
#
# Top Features (by SHAP value):
#   1. feature_1: 0.4567
#   2. feature_2: 0.3456
#   3. feature_3: 0.2345
#   ...
#
# Permutation Importance:
#   1. feature_1: 0.0543
#   2. feature_2: 0.0432
#   3. feature_3: 0.0321
#   ...
```

### Automated Insights

Generate insights from multiple experiments:

```bash
# Basic insights generation
exp-insights <run_id1>,<run_id2>,<run_id3>

# Use different metric
exp-insights <run_ids> --metric val.mae

# Group by parameter
exp-insights <run_ids> --group-by params.learning_rate

# Custom output directory
exp-insights <run_ids> --output-dir results/insights/

# Require minimum sample size
exp-insights <run_ids> --min-sample-size 10

# Verbose output
exp-insights <run_ids> --verbose
```

**Output:**
- Statistical test results (p-values, effect sizes)
- Automated recommendation
- Hyperparameter correlations
- Experiment rankings
- Summary statistics

**Example:**
```bash
# Generate insights from multiple runs
exp-insights abc123,def456,ghi789 --metric val.rmse --verbose

# Output:
# Analyzing 3 runs...
# Computing hyperparameter correlations...
# Ranking experiments...
#
# ============================================================
# Automated Insights Summary
# ============================================================
# Analyzed 3 runs
# Metric: val.rmse
#
# Best Run:
#   Run ID: def456
#
# Summary Statistics:
#   Mean: 15.2345
#   Std: 2.3456
#   Min: 12.3456
#   Max: 18.1234
#
# Statistical Tests:
#   t-test:
#     p-value: 0.0234
#     effect size: 0.8765
#
# Recommendation:
#   The best performing configuration shows statistically significant
#   improvement (large effect size: 0.88). Recommended for deployment.
#
# Top Hyperparameter Correlations:
#   learning_rate: 0.654
#   batch_size: -0.432
#   hidden_units: 0.321
#
# Top Experiments:
#   def456: 0.8765
#   ghi789: 0.7654
#   abc123: 0.6543
```

## Python API Usage

### Error Analysis

```python
from mlflow_tracking.analytics import ErrorAnalyzer
from mlflow_tracking.analytics.reporting import ReportGenerator

# Initialize analyzer
analyzer = ErrorAnalyzer()

# Load predictions from MLflow run
analyzer.load_run(run_id="abc123", predictions_path="predictions.csv")

# Compute residuals
residuals_df = analyzer.compute_residuals()

# Get error statistics
stats = analyzer.get_error_statistics()
print(f"Mean Absolute Error: {stats['mean_abs_error']:.4f}")
print(f"Median Absolute Error: {stats['median_abs_error']:.4f}")

# Identify failure modes
failure_modes = analyzer.identify_failure_modes(n_clusters=3)
print(failure_modes.head())

# Generate plots
from mlflow_tracking.analytics.visualizations import (
    plot_residuals,
    plot_error_distribution,
    plot_prediction_vs_actual,
)

residual_fig = plot_residuals(analyzer.predictions_df)
distribution_fig = plot_error_distribution(analyzer.predictions_df)
pred_vs_actual_fig = plot_prediction_vs_actual(analyzer.predictions_df)

# Save plots
residual_fig.savefig("residuals.png", dpi=150, bbox_inches='tight')
distribution_fig.savefig("error_distribution.png", dpi=150, bbox_inches='tight')
pred_vs_actual_fig.savefig("prediction_vs_actual.png", dpi=150, bbox_inches='tight')

# Generate HTML report
generator = ReportGenerator()
report_path = generator.generate_error_analysis_report(
    run_id="abc123",
    error_stats=stats,
    plot_paths={
        "residuals_plot": "residuals.png",
        "error_distribution_plot": "error_distribution.png",
        "prediction_vs_actual_plot": "prediction_vs_actual.png"
    },
    failure_modes=failure_modes,
    output_path="error_analysis_report.html"
)
print(f"Report saved to: {report_path}")
```

### Model Interpretability

```python
from mlflow_tracking.analytics import ModelInterpretability
from mlflow_tracking.analytics.reporting import ReportGenerator
import pandas as pd

# Initialize interpreter
interpreter = ModelInterpretability()

# Load model from MLflow
model = interpreter._load_model_from_artifacts(run_id="abc123")

# Load test data (you need to provide this)
X_test = pd.read_csv("test_features.csv")
y_test = pd.read_csv("test_labels.csv")

# Compute SHAP values
shap_values, explainer = interpreter.compute_shap(
    run_id="abc123",
    X_test=X_test
)

# Generate feature importance plot
importance_fig = interpreter.plot_feature_importance(
    shap_values,
    X_test,
    plot_type="summary",
    max_features=20
)

# Generate local explanation
local_fig = interpreter.plot_local_explanation(
    shap_values,
    X_test,
    sample_idx=0
)

# Compute permutation importance
perm_importance = interpreter.compute_permutation_importance(
    run_id="abc123",
    X_test=X_test,
    y_test=y_test
)

# Generate HTML report
generator = ReportGenerator()
report_path = generator.generate_interpretability_report(
    run_id="abc123",
    feature_importance=feature_importance_df,
    shap_summary_path="shap_summary.png",
    local_explanation_path="local_explanation.png",
    permutation_importance=perm_importance,
    output_path="interpretability_report.html"
)
print(f"Report saved to: {report_path}")
```

### Automated Insights

```python
from mlflow_tracking.analytics import InsightsGenerator
from mlflow_tracking.analytics.reporting import ReportGenerator
import json

# Initialize generator
generator = InsightsGenerator()

# Define run IDs to analyze
run_ids = ["abc123", "def456", "ghi789"]

# Generate insights
insights = generator.generate_insights(
    run_ids,
    metric="val.rmse",
    group_by="params.learning_rate",
    min_sample_size=5
)

# Check if sufficient data
if insights.get("status") == "insufficient_data":
    print(f"Warning: {insights['message']}")
    print(f"Recommendation: {insights['recommendation']}")
else:
    print(f"Best run: {insights['best_run']}")
    print(f"Recommendation: {insights['recommendation']}")

    # Get statistical tests
    if "statistical_tests" in insights:
        for test_name, test_result in insights["statistical_tests"].items():
            print(f"{test_name}: p-value={test_result['p_value']:.4f}")

    # Compute hyperparameter correlations
    correlations = generator.compare_hyperparameters(run_ids, metric="val.rmse")
    print(correlations.head())

    # Rank experiments
    rankings = generator.rank_experiments(run_ids)
    print(rankings.head())

    # Generate HTML report
    report_gen = ReportGenerator()
    report_path = report_gen.generate_insights_report(
        run_ids=run_ids,
        insights=insights,
        correlations=correlations,
        rankings=rankings,
        output_path="insights_report.html"
    )
    print(f"Report saved to: {report_path}")

    # Save JSON results
    with open("insights.json", "w") as f:
        json.dump(insights, f, indent=2, default=str)
```

### Report Generation

```python
from mlflow_tracking.analytics.reporting import ReportGenerator

# Initialize generator
generator = ReportGenerator()

# Generate individual reports
error_report = generator.generate_error_analysis_report(
    run_id="abc123",
    error_stats=stats,
    plot_paths=plot_paths,
    failure_modes=failure_modes,
    output_path="error_analysis_report.html"
)

interpretability_report = generator.generate_interpretability_report(
    run_id="abc123",
    feature_importance=feature_importance_df,
    shap_summary_path="shap_summary.png",
    local_explanation_path="local_explanation.png",
    permutation_importance=perm_importance,
    output_path="interpretability_report.html"
)

insights_report = generator.generate_insights_report(
    run_ids=["abc123", "def456"],
    insights=insights,
    correlations=correlations,
    rankings=rankings,
    output_path="insights_report.html"
)

# Generate comprehensive report (combines all)
comprehensive_report = generator.generate_comprehensive_report(
    run_ids=["abc123", "def456"],
    error_analysis=error_analysis_results,
    interpretability=interpretability_results,
    insights=insights_results,
    output_format="html",
    output_path="comprehensive_report"
)

# Convert HTML to PDF (requires WeasyPrint)
try:
    pdf_report = generator.convert_html_to_pdf(
        html_path="comprehensive_report.html",
        pdf_path="comprehensive_report.pdf"
    )
    print(f"PDF report saved to: {pdf_report}")
except ImportError:
    print("WeasyPrint not installed. Install with: pip install weasyprint")
```

## Workflow Examples

### Example 1: Analyze Prediction Errors for a Single Run

```bash
# 1. Run your experiment and log predictions
exp-run config.yaml

# 2. Analyze errors
exp-analyze-errors <run_id> --verbose

# 3. Review the generated plots and statistics
# 4. Identify failure modes and investigate systematic errors
```

**Python Equivalent:**
```python
from mlflow_tracking.analytics import ErrorAnalyzer

analyzer = ErrorAnalyzer()
analyzer.load_run(run_id, "predictions.csv")
stats = analyzer.get_error_statistics()
failure_modes = analyzer.identify_failure_modes(n_clusters=3)

print(f"Mean Absolute Error: {stats['mean_abs_error']:.4f}")
print(f"Failure Modes: {len(failure_modes['cluster'].unique())} clusters identified")
```

### Example 2: Compare Feature Importance Across Multiple Models

```bash
# 1. Train multiple models with different hyperparameters
exp-run config1.yaml
exp-run config2.yaml
exp-run config3.yaml

# 2. Generate interpretability reports for each
exp-interpret <run_id1> --output-dir results/model1/
exp-interpret <run_id2> --output-dir results/model2/
exp-interpret <run_id3> --output-dir results/model3/

# 3. Compare feature importance rankings
# 4. Identify which features are consistently important
```

**Python Equivalent:**
```python
from mlflow_tracking.analytics import ModelInterpretability

run_ids = ["run1", "run2", "run3"]
importance rankings = {}

for run_id in run_ids:
    interpreter = ModelInterpretability()
    model = interpreter._load_model_from_artifacts(run_id)
    shap_values, _ = interpreter.compute_shap(run_id, X_test)
    rankings[run_id] = interpreter.get_feature_importance(shap_values)

# Compare rankings across models
```

### Example 3: Generate Insights from Hyperparameter Optimization

```bash
# 1. Run hyperparameter optimization
exp-run-optimize optimization_config.yaml --n-trials 50

# 2. Collect all run IDs from the optimization study
# 3. Generate insights across all trials
exp-insights <run_id1>,<run_id2>,...,<run_id50> --metric val.rmse --verbose

# 4. Review statistical tests and correlations
# 5. Use recommendations to guide further optimization
```

**Python Equivalent:**
```python
from mlflow_tracking.analytics import InsightsGenerator

# Get all run IDs from optimization study
run_ids = get_optimization_run_ids(study_name="my_study")

# Generate insights
generator = InsightsGenerator()
insights = generator.generate_insights(run_ids, metric="val.rmse")

# Get correlations
correlations = generator.compare_hyperparameters(run_ids, metric="val.rmse")

# Print top correlations
print(correlations.head(10))
```

### Example 4: Create Comprehensive Report for Model Selection

```python
from mlflow_tracking.analytics import (
    ErrorAnalyzer,
    ModelInterpretability,
    InsightsGenerator
)
from mlflow_tracking.analytics.reporting import ReportGenerator

# 1. Analyze candidates
candidate_runs = ["model1", "model2", "model3"]

# 2. Generate insights and rank models
generator = InsightsGenerator()
rankings = generator.rank_experiments(candidate_runs)
best_run = rankings.iloc[0]['run_id']

# 3. Deep dive into best model
analyzer = ErrorAnalyzer()
analyzer.load_run(best_run)
error_stats = analyzer.get_error_statistics()

# 4. Generate interpretability analysis
interpreter = ModelInterpretability()
shap_values, explainer = interpreter.compute_shap(best_run, X_test)

# 5. Create comprehensive report
report_gen = ReportGenerator()
report_gen.generate_comprehensive_report(
    run_ids=candidate_runs,
    error_analysis={"run_id": best_run, "stats": error_stats},
    interpretability={"run_id": best_run, "shap_values": shap_values},
    insights={"rankings": rankings},
    output_format="pdf",
    output_path="model_selection_report"
)

print(f"Best model: {best_run}")
print(f"Report saved to: model_selection_report.pdf")
```

## Interpreting Results

### How to Read Residual Plots

Residual plots show the difference between predicted and actual values:

- **Good model**: Residuals randomly scattered around zero with constant variance
- **Heteroscedasticity**: Residuals spread increases/decreases with predictions (non-constant variance)
- **Bias**: Systematic pattern (e.g., curve) indicates model misspecification
- **Outliers**: Points far from zero indicate prediction errors

**Key Patterns:**
- Random scatter: Model is well-specified
- Funnel shape: Heteroscedasticity (consider log transformation)
- U-shape: Non-linear relationship not captured
- Clusters: Subgroups in data with different characteristics

### How to Interpret SHAP Values

SHAP (SHapley Additive exPlanations) values show feature contributions:

- **Positive SHAP value**: Feature pushes prediction higher
- **Negative SHAP value**: Feature pushes prediction lower
- **Absolute value**: Magnitude of importance
- **Color**: Feature value (red = high, blue = low)

**Summary Plot:**
- Y-axis: Features ranked by importance
- X-axis: SHAP value (impact on prediction)
- Color: Feature value
- Wide distribution = high impact feature

**Local Explanation:**
- Shows contribution of each feature to single prediction
- Base value: Average prediction across training data
- Final prediction = base value + sum of SHAP values

### How to Understand Statistical Test Results

Statistical tests compare groups to determine if differences are significant:

**p-value:**
- < 0.05: Statistically significant (reject null hypothesis)
- >= 0.05: Not statistically significant (cannot reject null hypothesis)
- Lower = more significant

**Effect Size (Cohen's d):**
- < 0.2: Negligible
- 0.2 - 0.5: Small
- 0.5 - 0.8: Medium
- >= 0.8: Large

**Interpretation:**
- Significant + Large effect: Strong evidence of difference
- Significant + Small effect: Statistically significant but may not be practically important
- Not significant: Cannot conclude difference exists (may need more data)

**Automatic Test Selection:**
- Normality test (Shapiro-Wilk) is performed first
- If normal: Use t-test
- If not normal: Use Mann-Whitney U test

### How to Use Recommendations

The automated recommendations synthesize statistical results into actionable advice:

**Example Recommendations:**
- "No significant difference found. Collect more data or refine experimental design."
- "Significant but negligible effect. Consider practical significance before deployment."
- "Significant small effect. Consider cost-benefit trade-offs."
- "Significant medium effect. Recommended for deployment."
- "Significant large effect. Strongly recommended for deployment."

**Using Recommendations:**
1. Check p-value: Is the result statistically significant?
2. Check effect size: Is the difference practically meaningful?
3. Consider context: Domain knowledge and business impact
4. Take action: Deploy, collect more data, or refine design

## Best Practices

### 1. Always Log Analysis as MLflow Artifacts

```bash
# Good: Log to MLflow for reproducibility
exp-analyze-errors <run_id> --verbose

# Store analysis with experiment data
# - Traceability
# - Reproducibility
# - Collaboration
```

### 2. Use Appropriate Sample Sizes for Statistical Testing

```python
# Too few samples: Low power, unreliable results
insights = generator.generate_insights(
    run_ids=["run1", "run2"],  # Only 2 runs
    min_sample_size=5  # Will warn about insufficient data
)

# Good: Sufficient samples for reliable statistics
insights = generator.generate_insights(
    run_ids=list_of_20_runs,  # 20 runs
    min_sample_size=5  # Sufficient for analysis
)
```

### 3. Validate Assumptions Before Interpreting Tests

```python
# Check normality assumption
# - If met: t-test is appropriate
# - If not: Mann-Whitney U test is used automatically

# InsightsGenerator handles this automatically
insights = generator.generate_insights(run_ids)
# Automatic test selection based on Shapiro-Wilk test
```

### 4. Combine Multiple Analyses for Comprehensive Understanding

```python
# Don't rely on single metric
# - Error analysis: What are the failure modes?
# - Interpretability: Which features drive predictions?
# - Insights: How do hyperparameters affect performance?

# Use all three for complete picture
```

### 5. Visualize Results Before Making Decisions

```bash
# Generate plots
exp-analyze-errors <run_id> --verbose

# Review plots before proceeding
# - Check for patterns in residuals
# - Identify outliers
# - Understand error distribution
```

## Troubleshooting

### Common Errors and Solutions

**Error: "No module named 'shap'"**
```bash
# Solution: Install SHAP
pip install shap
```

**Error: "predictions.csv not found in artifacts"**
```bash
# Solution: Ensure predictions are logged during experiment
# In your training script:
tracker.log_artifact("predictions.csv")
```

**Error: "Insufficient data for statistical testing"**
```bash
# Solution: Provide more runs or reduce min_sample_size
exp-insights <run_ids> --min-sample-size 3
```

**Error: "Model artifact not found"**
```bash
# Solution: Ensure model is logged during experiment
# In your training script:
mlflow.sklearn.log_model(model, "model")
```

### What to Do If SHAP Computation is Slow

SHAP computation can be slow for large datasets:

```python
# Solution 1: Use subset of data
shap_values, explainer = interpreter.compute_shap(
    run_id="abc123",
    X_test=X_test.sample(n=1000, random_state=42)  # Use 1000 samples
)

# Solution 2: Use fewer background samples
explainer = shap.Explainer(model, X_background.sample(n=100))

# Solution 3: Use faster approximation methods
explainer = shap.Explainer(model, X_background, algorithm="auto")
```

### How to Handle Missing Predictions Artifacts

If predictions artifact doesn't exist:

```bash
# Solution: Create predictions artifact during training
# In your training script:
import pandas as pd

predictions = model.predict(X_test)
predictions_df = pd.DataFrame({
    'image_id': test_ids,
    'actual': y_test,
    'predicted': predictions
})
predictions_df.to_csv("predictions.csv", index=False)

tracker.log_artifact("predictions.csv")
```

### ELI5 Installation Issues

ELI5 may have installation issues on some systems:

```bash
# Solution 1: Install with pip
pip install eli5

# Solution 2: Use SHAP instead (more reliable)
# ELI5 is optional; SHAP provides similar functionality

# Solution 3: Skip permutation importance
exp-interpret <run_id>  # Without --compute-permutation flag
```

### WeasyPrint Installation Issues

WeasyPrint can be difficult to install on macOS:

```bash
# Solution 1: Use Homebrew
brew install python-tk python@3.9
pip install weasyprint

# Solution 2: Skip PDF generation (HTML still works)
# Use HTML reports instead of PDF

# Solution 3: Use wkhtmltopdf as alternative
brew install wkhtmltopdf
```

## Additional Resources

### Research Papers

- **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
  - https://arxiv.org/abs/1705.07874

- **Permutation Importance**: Altmann, A., et al. (2010). Permutation importance: a corrected feature importance measure. Bioinformatics.
  - https://academic.oup.com/bioinformatics/article/26/10/1340/193345

- **Effect Size**: Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Routledge.

### Documentation

- **SHAP Documentation**: https://shap.readthedocs.io/
- **ELI5 Documentation**: https://eli5.readthedocs.io/
- **Jinja2 Documentation**: https://jinja.palletsprojects.com/
- **WeasyPrint Documentation**: https://weasyprint.readthedocs.io/

### Examples

See the `examples/configs/analytics/` directory for:
- Sample Jupyter notebooks
- Example scripts
- Sample analysis results

## Support

For issues or questions:
1. Check this README for common solutions
2. Review error messages carefully
3. Consult the research papers for theoretical background
4. Check library documentation (SHAP, ELI5, etc.)
5. Open an issue on the project repository

---

**Generated by MLflow Analytics - Advanced Analytics Module**

Last updated: 2025-01-17
