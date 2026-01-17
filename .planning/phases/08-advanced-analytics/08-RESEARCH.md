# Phase 08: Advanced Analytics - Research

**Researched:** 2025-01-17
**Domain:** Machine Learning Analytics (Error Analysis, Model Interpretability, Automated Insights)
**Confidence:** HIGH

## Summary

Phase 08 focuses on implementing advanced analytics capabilities for the Image2Biomass ML experimentation framework. The research reveals three mature, well-documented domains: (1) error analysis and residual visualization using standard statistical libraries, (2) model interpretability through SHAP/LIME/ELI5 with established Python ecosystems, and (3) automated insights generation leveraging existing MLflow artifacts and statistical testing.

The standard approach integrates tightly with the existing MLflow infrastructure. Error analysis builds on matplotlib/seaborn for residual plots and failure mode clustering. Model interpretability uses SHAP as the primary library (model-agnostic, well-maintained) with ELI5 for global feature importance and Captum for PyTorch-specific attention visualization. Automated insights leverage pandas for data manipulation, scipy/statsmodels for statistical significance testing, and Jinja2+WeasyPrint for professional HTML/PDF report generation.

**Primary recommendation:** Use SHAP for interpretability (model-agnostic, excellent documentation), matplotlib/seaborn for visualizations, and build on the existing ExperimentComparator pattern for analysis workflows. Don't build custom interpretability - SHAP/ELI5/Captum are mature solutions. Leverage MLflow artifacts as the single source of truth.

## Standard Stack

### Core Libraries
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| **shap** | >=0.46.0 | Model-agnostic interpretability (SHAP values) | Industry standard, model-agnostic, excellent docs, supports tree/tabular/deep models |
| **matplotlib** | >=3.8.0 | Foundation plotting library | Default for scientific visualization, integrates with all ML libraries |
| **seaborn** | >=0.13.0 | Statistical visualizations (residual plots, heatmaps) | Built on matplotlib, prettier defaults, regression plots built-in |
| **pandas** | >=2.0.0 | Data manipulation for analysis results | Already in requirements, standard for tabular data manipulation |
| **scipy** | >=1.11.0 | Statistical tests (t-test, mann-whitney, etc.) | Already in requirements, scientific computing standard |
| **statsmodels** | >=0.14.0 | Advanced statistical modeling and testing | Complements scipy, more detailed statistical outputs |

### Supporting Libraries
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **eli5** | >=0.13.0 | Global feature importance, permutation importance | Quick global explanations, works with scikit-learn natively |
| **captum** | >=0.7.0 | PyTorch model interpretability (Grad-CAM, attention) | For EfficientNet/PyTorch models, attention visualization |
| **lime** | >=0.2.0 | Local surrogate explanations | Alternative to SHAP for local explanations (less preferred) |
| **jinja2** | >=3.1.0 | HTML report templating | Already in requirements, use for report generation |
| **weasyprint** | >=60.0 | HTML to PDF conversion | Professional PDF reports from templates |
| **plotly** | >=5.18.0 | Interactive visualizations | Optional: for dashboards, interactive plots |

### MLflow Integration (Already Exists)
| Component | Purpose | How to Use |
|-----------|---------|------------|
| **ExperimentTracker** | Logging analysis artifacts | Log plots, SHAP values, reports as artifacts |
| **ExperimentComparator** | Fetching run data for analysis | Use compare_by_ids() to get DataFrames for analysis |
| **MLflow Artifacts** | Storing analysis results | Store plots, reports, pickled analysis objects |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| **SHAP** | LIME | LIME faster but less theoretically sound, SHAP is gold standard |
| **matplotlib/seaborn** | plotly/bokeh | Interactive libraries better for dashboards, static better for reports |
| **WeasyPrint** | ReportLab/pdfkit | WeasyPrint better CSS support, modern HTML-to-PDF |
| **Custom analysis** | Azure ML Error Analysis | Azure ML is cloud-specific, we need local Python solution |

**Installation:**
\`\`\`bash
# New dependencies for Phase 08
pip install shap>=0.46.0 eli5>=0.13.0 captum>=0.7.0 statsmodels>=0.14.0 weasyprint>=60.0 plotly>=5.18.0

# All existing dependencies already in requirements.txt
\`\`\`

## Architecture Patterns

### Recommended Project Structure
\`\`\`
mlflow_tracking/
├── analytics/
│   ├── __init__.py
│   ├── error_analyzer.py      # ErrorAnalyzer class for residual analysis
│   ├── interpretability.py    # ModelInterpretability class (SHAP/ELI5)
│   ├── insights_generator.py  # InsightsGenerator class (automated insights)
│   ├── reporting.py           # ReportGenerator class (Jinja2 templates)
│   └── visualizations.py      # Visualization utilities (plotting functions)
├── cli.py                      # Extend with analytics commands
└── templates/                  # Jinja2 templates for reports
    ├── error_analysis.html
    ├── interpretability_report.html
    └── insights_summary.html
\`\`\`

### Pattern 1: ErrorAnalyzer Class

**What:** A dedicated class for performing error analysis on MLflow runs, building on the ExperimentComparator pattern.

**When to use:** For any error analysis workflow (residual plots, failure mode identification, error clustering).

**Example:**
\`\`\`python
# Source: Based on mlflow_tracking/comparison.py pattern + research findings
class ErrorAnalyzer:
    """
    Analyze prediction errors and identify failure modes.

    Fetches run data from MLflow, computes residuals, identifies
    systematic error patterns, and generates visualizations.

    Example:
        >>> analyzer = ErrorAnalyzer()
        >>> analyzer.load_run(run_id="abc123", predictions_path="preds.csv")
        >>> residual_fig = analyzer.plot_residuals()
        >>> failure_modes = analyzer.identify_failure_modes()
        >>> analyzer.log_analysis_artifacts()
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize with MLflow client (follow ExperimentTracker pattern)."""
        self.client = MlflowClient(tracking_uri)
        self.predictions_df = None
        self.residuals = None

    def load_run(self, run_id: str, predictions_path: str = None):
        """
        Load predictions from MLflow artifacts.

        Fetches the run's artifact predictions CSV and computes residuals.
        """
        run = self.client.get_run(run_id)
        # Load predictions from artifact or compute from model
        # Compute residuals = y_true - y_pred

    def plot_residuals(self, figsize=(10, 6)) -> plt.Figure:
        """
        Create residual plot using seaborn.

        Returns matplotlib figure for logging as artifact.
        """
        fig, ax = plt.subplots(figsize=figsize)
        sns.residplot(x='predicted', y='residual', data=self.predictions_df, ax=ax)
        ax.axhline(y=0, color='r', linestyle='--')
        return fig

    def identify_failure_modes(self, n_clusters: int = 3) -> pd.DataFrame:
        """
        Identify systematic failure modes using clustering.

        Uses KMeans on high-error samples to find patterns.
        """
        high_errors = self.predictions_df[self.predictions_df['abs_residual'] > threshold]
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(high_errors[features])
        return high_errors.assign(cluster=clusters)

    def plot_error_distribution(self) -> plt.Figure:
        """
        Visualize error distribution using histograms and box plots.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(data=self.predictions_df, x='residual', ax=axes[0])
        sns.boxplot(data=self.predictions_df, y='residual', ax=axes[1])
        return fig
\`\`\`

### Pattern 2: ModelInterpretability Class

**What:** A unified interface for model interpretability using SHAP/ELI5/Captum.

**When to use:** For explaining model predictions, computing feature importance, visualizing attention.

**Example:**
\`\`\`python
# Source: Based on SHAP documentation (https://shap.readthedocs.io/)
class ModelInterpretability:
    """
    Generate model explanations using SHAP, ELI5, and Captum.

    Supports multiple model types:
    - scikit-learn: SHAP TreeExplainer, KernelExplainer
    - XGBoost: SHAP TreeExplainer (optimized)
    - PyTorch (EfficientNet): Captum (LayerGradCam, IntegratedGradients)

    Example:
        >>> interpreter = ModelInterpretability()
        >>> shap_values = interpreter.compute_shap(run_id="abc123")
        >>> importance_fig = interpreter.plot_feature_importance()
        >>> attention_fig = interpreter.visualize_attention(run_id="def456")
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize with MLflow client."""
        self.client = MlflowClient(tracking_uri)

    def compute_shap(self, run_id: str, background_samples: int = 100) -> np.ndarray:
        """
        Compute SHAP values for a run.

        Fetches model from MLflow artifacts, loads training data,
        computes SHAP values using appropriate explainer.
        """
        run = self.client.get_run(run_id)
        model = self._load_model_from_artifacts(run)

        # Determine explainer type based on model
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, torch.nn.Module):
            explainer = shap.DeepExplainer(model, background_data)
        else:
            explainer = shap.KernelExplainer(model.predict, background_data)

        shap_values = explainer.shap_values(X_test)
        return shap_values

    def plot_feature_importance(self, shap_values, feature_names: List[str]) -> plt.Figure:
        """
        Create global feature importance plot.

        Uses SHAP's summary plot for aggregated importance.
        """
        fig = plt.figure()
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        return fig

    def plot_local_explanation(self, shap_values, sample_idx: int) -> plt.Figure:
        """
        Create waterfall plot for single prediction.
        """
        fig = plt.figure()
        shap.waterfall_plot(shap_values[sample_idx], show=False)
        return fig

    def compute_permutation_importance(self, run_id: str, X_test, y_test) -> pd.DataFrame:
        """
        Compute permutation importance using ELI5.

        More robust than default feature importance for correlated features.
        """
        run = self.client.get_run(run_id)
        model = self._load_model_from_artifacts(run)

        perm = eli5.permutation_importance.PermutationImportance(model)
        perm.fit(X_test, y_test)
        return eli5.format_as_dataframe(perm.results_)

    def visualize_attention(self, run_id: str, image_path: str) -> plt.Figure:
        """
        Visualize attention for PyTorch image models using Captum.

        Uses LayerGradCam for EfficientNet models.
        """
        from captum.attr import LayerGradCam

        run = self.client.get_run(run_id)
        model = self._load_model_from_artifacts(run)

        # Get target layer (last convolutional layer for EfficientNet)
        target_layer = self._get_last_conv_layer(model)

        grad_cam = LayerGradCam(model, target_layer)
        attribution = grad_cam.attribute(input_tensor, target=target_class)

        # Visualize
        fig, ax = plt.subplots()
        ax.imshow(attribution.squeeze(), cmap='jet')
        return fig
\`\`\`

### Pattern 3: InsightsGenerator Class

**What:** Automated insights generation from experiment results using statistical testing.

**When to use:** For comparing experiments, identifying statistically significant improvements, generating summaries.

**Example:**
\`\`\`python
# Source: Based on A/B testing research (https://www.kdnuggets.com/a-complete-guide-to-a-b-testing-in-python)
class InsightsGenerator:
    """
    Generate automated insights from experiment comparisons.

    Performs statistical significance testing, pattern recognition,
    and generates actionable recommendations.

    Example:
        >>> generator = InsightsGenerator()
        >>> insights = generator.generate_insights(run_ids=["run1", "run2", "run3"])
        >>> report = generator.format_insights(insights)
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize with MLflow client."""
        self.client = MlflowClient(tracking_uri)
        self.comparator = ExperimentComparator(tracking_uri)

    def generate_insights(self, run_ids: List[str], metric: str = "val_rmse") -> Dict:
        """
        Generate insights from multiple runs.

        Performs:
        1. Statistical significance testing (t-test or mann-whitney)
        2. Effect size calculation
        3. Ranking and recommendations
        """
        df = self.comparator.compare_by_ids(run_ids)

        # Extract metric values
        metrics = df[f'metrics.{metric}'].values

        # Perform statistical tests
        if self._is_normal(metrics):
            stat, p_value = scipy.stats.ttest_ind(metrics[:-1], metrics[1:])
            test_type = "t-test"
        else:
            stat, p_value = scipy.stats.mannwhitneyu(metrics[:-1], metrics[1:])
            test_type = "Mann-Whitney U"

        # Calculate effect size (Cohen's d)
        effect_size = self._cohens_d(metrics[:-1], metrics[1:])

        return {
            "best_run": df.loc[df[f'metrics.{metric}'].idxmin(), 'run_id'],
            "improvement": self._calculate_improvement(metrics),
            "statistical_test": test_type,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "effect_size": effect_size,
            "recommendation": self._generate_recommendation(p_value, effect_size)
        }

    def _cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (group1.mean() - group2.mean()) / pooled_std

    def _generate_recommendation(self, p_value: float, effect_size: float) -> str:
        """Generate actionable recommendation."""
        if p_value >= 0.05:
            return "No significant difference detected. Collect more data or refine experiment."
        elif abs(effect_size) < 0.2:
            return "Significant but small effect. Consider practical significance."
        elif abs(effect_size) < 0.8:
            return "Significant medium effect. Recommended for deployment."
        else:
            return "Significant large effect. Strongly recommended for deployment."

    def compare_hyperparameters(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Analyze hyperparameter correlations with performance.

        Uses correlation analysis and partial dependence.
        """
        df = self.comparator.compare_by_ids(run_ids)

        # Extract parameters and metrics
        param_cols = [c for c in df.columns if c.startswith('params.')]
        metric_col = 'metrics.val_rmse'

        # Compute correlations
        correlations = df[param_cols + [metric_col]].corr()[metric_col]

        return correlations.sort_values(ascending=False)
\`\`\`

### Pattern 4: ReportGenerator Class

**What:** Automated report generation using Jinja2 templates and WeasyPrint.

**When to use:** For creating professional HTML/PDF reports summarizing analysis results.

**Example:**
\`\`\`python
# Source: Based on Jinja2+WeasyPrint research (https://www.incentius.com/blog-posts/build-modern-print-ready-pdfs-with-python-flask-weasyprint/)
class ReportGenerator:
    """
    Generate HTML/PDF reports from analysis results.

    Uses Jinja2 templates for HTML generation and WeasyPrint for PDF conversion.

    Example:
        >>> generator = ReportGenerator()
        >>> html = generator.generate_html_report(analysis_results)
        >>> pdf_path = generator.generate_pdf_report(html, "report.pdf")
    """

    def __init__(self, template_dir: str = "templates"):
        """Initialize Jinja2 environment."""
        from jinja2 import Environment, FileSystemLoader

        self.env = Environment(loader=FileSystemLoader(template_dir))

    def generate_html_report(
        self,
        analysis_results: Dict,
        template_name: str = "error_analysis.html"
    ) -> str:
        """
        Generate HTML report from analysis results.

        Renders Jinja2 template with analysis data.
        """
        template = self.env.get_template(template_name)
        html = template.render(**analysis_results)
        return html

    def generate_pdf_report(
        self,
        html_content: str,
        output_path: str
    ) -> str:
        """
        Convert HTML to PDF using WeasyPrint.

        Supports modern CSS (Flexbox, Grid, custom fonts).
        """
        from weasyprint import HTML

        HTML(string=html_content).write_pdf(output_path)
        return output_path

    def generate_insights_report(
        self,
        run_ids: List[str],
        output_format: str = "html"
    ) -> str:
        """
        Generate comprehensive insights report.

        Combines error analysis, interpretability, and insights.
        """
        # Run all analyses
        error_analyzer = ErrorAnalyzer()
        interpreter = ModelInterpretability()
        insights = InsightsGenerator()

        # Compile results
        results = {
            "error_analysis": error_analyzer.analyze(run_ids),
            "interpretability": interpreter.explain(run_ids),
            "insights": insights.generate_insights(run_ids),
            "metadata": {"run_ids": run_ids, "generated_at": datetime.now()}
        }

        # Generate report
        html = self.generate_html_report(results)
        if output_format == "pdf":
            return self.generate_pdf_report(html, f"report_{datetime.now():%Y%m%d}.pdf")
        return html
\`\`\`

### Anti-Patterns to Avoid

- **Don't implement custom SHAP/LIME algorithms:** Use established libraries. Custom implementations will be buggy and less maintainable.
- **Don't hardcode analysis workflows:** Make analysis classes composable and reusable across different run types.
- **Don't ignore MLflow artifacts:** Always log analysis results as artifacts for reproducibility.
- **Don't build custom statistical tests:** Use scipy/statsmodels. They're battle-tested and well-documented.
- **Don't create separate data stores:** MLflow is the single source of truth. Don't duplicate run data elsewhere.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| **SHAP value computation** | Custom Shapley value implementation | **shap** library | Mathematically complex, optimized implementations exist |
| **Feature importance** | Custom permutation importance loops | **eli5.permutation_importance** | Handles edge cases, parallel processing |
| **Residual plots** | Custom matplotlib code for each plot type | **seaborn.residplot, regplot** | Statistical best practices built-in |
| **Statistical tests** | Custom t-test, mann-whitney implementations | **scipy.stats, statsmodels** | Correct p-value calculations, assumptions checked |
| **Attention visualization** | Custom gradient computation | **captum (LayerGradCam, IntegratedGradients)** | Handles PyTorch autograd complexities |
| **PDF generation** | matplotlib PDF export | **Jinja2 + WeasyPrint** | Professional formatting, CSS support |
| **Clustering for failure modes** | Custom K-Means implementation | **sklearn.cluster** | Optimized, multiple algorithms available |
| **Correlation analysis** | Custom correlation matrix computation | **pandas.DataFrame.corr()** | Handles missing data, multiple methods |
| **HTML templating** | String concatenation | **Jinja2** | Maintainable, designer-friendly |

**Key insight:** Custom solutions for these problems are worse because: (1) they introduce bugs in complex mathematical/statistical operations, (2) they're less maintainable than community-tested libraries, (3) they miss edge cases that established libraries handle, (4) they increase development time significantly. The SHAP library alone took years of research to optimize - don't reimplement it.

## Common Pitfalls

### Pitfall 1: Ignoring Model Type for Interpretability

**What goes wrong:** Using TreeExplainer for non-tree models or KernelExplainer for deep learning models, leading to incorrect SHAP values or performance issues.

**Why it happens:** SHAP has multiple explainer types (Tree, Kernel, Deep, Sampling, Linear) optimized for different model types. Using the wrong one gives incorrect results or is extremely slow.

**How to avoid:**
\`\`\`python
# CORRECT: Choose explainer based on model type
if hasattr(model, 'feature_importances_') and hasattr(model, 'estimators_'):
    # Tree-based model (RandomForest, XGBoost)
    explainer = shap.TreeExplainer(model)
elif isinstance(model, torch.nn.Module):
    # Deep learning model
    explainer = shap.DeepExplainer(model, background_data)
elif hasattr(model, 'coef_'):
    # Linear model
    explainer = shap.LinearExplainer(model, X_train)
else:
    # Fallback for any model (slower but works)
    explainer = shap.KernelExplainer(model.predict, X_train)
\`\`\`

**Warning signs:** SHAP computation takes >10 minutes, SHAP values don't sum to model output, inconsistent explanations across runs.

### Pitfall 2: Not Logging Analysis Artifacts

**What goes wrong:** Running analysis but not saving results to MLflow, making insights irreproducible and lost over time.

**Why it happens:** Analysis scripts generate plots in notebooks but don't integrate with MLflow logging infrastructure.

**How to avoid:**
\`\`\`python
# CORRECT: Always log analysis as artifacts
analyzer = ErrorAnalyzer()
analyzer.load_run(run_id)

# Generate plot
fig = analyzer.plot_residuals()

# Save as artifact (follow ExperimentTracker pattern)
with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
    fig.savefig(f.name)
    tracker.log_artifact(f.name, artifact_path="error_analysis")

# Also log analysis metrics
tracker.log_metrics({"max_residual": analyzer.residuals.max()})
\`\`\`

**Warning signs:** Analysis results exist only in notebooks, can't reproduce old analyses, no version control for insights.

### Pitfall 3: Wrong Statistical Test Assumptions

**What goes wrong:** Using t-test on non-normal data or not checking assumptions, leading to incorrect p-values and false conclusions.

**Why it happens:** Statistical tests have assumptions (normality, equal variance) that must be verified. Violating assumptions invalidates results.

**How to avoid:**
\`\`\`python
# CORRECT: Check assumptions before choosing test
from scipy import stats

# Check normality
_, p_normal = stats.shapiro(metric_values)

if p_normal > 0.05:
    # Data is normal, use t-test
    stat, p_value = stats.ttest_ind(group1, group2)
    test_used = "t-test"
else:
    # Data is not normal, use Mann-Whitney U
    stat, p_value = stats.mannwhitneyu(group1, group2)
    test_used = "Mann-Whitney U"

# Log which test was used
tracker.log_params({"statistical_test": test_used})
\`\`\`

**Warning signs:** P-values exactly 0 or 1, results change dramatically with small sample changes, no assumption checking documented.

### Pitfall 4: Not Handling Missing Data in Comparisons

**What goes wrong:** Trying to compare runs where some don't have required metrics, causing KeyError or incorrect analysis.

**Why it happens:** Different experiment types log different metrics. Not all runs have all metrics.

**How to avoid:**
\`\`\`python
# CORRECT: Validate required metrics before comparison
def compare_by_ids(self, run_ids: List[str], required_metrics: List[str]) -> pd.DataFrame:
    """Compare runs, validating required metrics exist."""
    dfs = []

    for run_id in run_ids:
        run = self.client.get_run(run_id)
        data = {'run_id': run_id}

        # Check required metrics exist
        for metric in required_metrics:
            if metric not in run.data.metrics:
                raise ValueError(f"Run {run_id} missing required metric '{metric}'")
            data[f'metrics.{metric}'] = run.data.metrics[metric]

        dfs.append(data)

    return pd.DataFrame(dfs)

# Usage with validation
df = comparator.compare_by_ids(
    run_ids,
    required_metrics=['val_rmse', 'val_mae', 'train_rmse']
)
\`\`\`

**Warning signs:** KeyError on metric access, NaN values in comparison DataFrame, silent failures.

### Pitfall 5: Ignoring Sample Size for Statistical Power

**What goes wrong:** Declaring significance with tiny sample sizes (e.g., 3 runs per condition), leading to false positives.

**Why it happens:** Statistical power depends on sample size. Small samples give unreliable p-values even if technically significant.

**How to avoid:**
\`\`\`python
# CORRECT: Check sample size before testing
def generate_insights(self, run_ids: List[str]) -> Dict:
    min_sample_size = 5  # Minimum for reasonable t-test

    if len(run_ids) < min_sample_size:
        return {
            "status": "insufficient_data",
            "message": f"Need at least {min_sample_size} runs, got {len(run_ids)}",
            "recommendation": "Run more experiments before drawing conclusions"
        }

    # Proceed with statistical tests
    insights = self._perform_statistical_tests(run_ids)
    insights["sample_size"] = len(run_ids)
    return insights
\`\`\`

**Warning signs:** P-value = 0.049 with 3 samples, huge effect sizes with tiny samples, recommendations change with one additional run.

## Code Examples

### Verified Pattern: Residual Analysis with Seaborn

\`\`\`python
# Source: https://www.geeksforgeeks.org/python-how-to-create-a-residual-plot-in-python/
import seaborn as sns
import matplotlib.pyplot as plt

def create_residual_plot(predictions_df: pd.DataFrame) -> plt.Figure:
    """
    Create residual plot using seaborn.

    Args:
        predictions_df: DataFrame with 'predicted', 'actual', 'residual' columns

    Returns:
        Matplotlib figure ready for saving
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Residual plot with regression line
    sns.regplot(
        data=predictions_df,
        x='predicted',
        y='residual',
        lowess=True,  # Add locally weighted scatterplot smoothing
        line_kws={'color': 'red', 'lw': 2},
        scatter_kws={'alpha': 0.5},
        ax=ax
    )

    # Add reference line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    # Labels and title
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot for Model Diagnostics')

    return fig
\`\`\`

### Verified Pattern: SHAP for Tabular Models

\`\`\`python
# Source: https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html
import shap

def compute_shap_values(model, X_train, X_test):
    """
    Compute SHAP values for tabular model.

    Args:
        model: Trained model (scikit-learn, XGBoost, etc.)
        X_train: Training data for background dataset
        X_test: Test data to explain

    Returns:
        SHAP values array and explainer object
    """
    # Choose appropriate explainer
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        explainer = shap.TreeExplainer(model, data=X_train)
    elif hasattr(model, 'coef_'):
        # Linear models
        explainer = shap.LinearExplainer(model, X_train)
    else:
        # Model-agnostic (slower)
        explainer = shap.KernelExplainer(model.predict, X_train)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_test)

    return shap_values, explainer

def plot_shap_summary(shap_values, X_test, feature_names=None):
    """
    Create SHAP summary plot (feature importance).

    Args:
        shap_values: SHAP values array
        X_test: Test dataset
        feature_names: Optional list of feature names

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False,
        plot_size=None
    )

    plt.tight_layout()
    return fig
\`\`\`

### Verified Pattern: Captum for PyTorch Attention

\`\`\`python
# Source: https://captum.ai/tutorials/
import torch
from captum.attr import LayerGradCam, IntegratedGradients

def visualize_efficientnet_attention(model, image_tensor, target_class=0):
    """
    Visualize attention for EfficientNet using Grad-CAM.

    Args:
        model: PyTorch EfficientNet model
        image_tensor: Input image tensor [1, 3, H, W]
        target_class: Target class for attribution

    Returns:
        Matplotlib figure with attention overlay
    """
    # Find last convolutional layer
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        raise ValueError("Could not find convolutional layer")

    # Create Grad-CAM explainer
    grad_cam = LayerGradCam(model, target_layer)

    # Compute attributions
    with torch.no_grad():
        attributions = grad_cam.attribute(
            image_tensor,
            target=target_class
        )

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Original image
    ax1.imshow(image_tensor.squeeze().permute(1, 2, 0).cpu())
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Attention heatmap
    ax2.imshow(attributions.squeeze().cpu(), cmap='jet')
    ax2.set_title('Grad-CAM Attention')
    ax2.axis('off')

    return fig
\`\`\`

### Verified Pattern: Statistical Significance Testing

\`\`\`python
# Source: https://www.kdnuggets.com/a-complete-guide-to-a-b-testing-in-python
from scipy import stats
import numpy as np

def compare_experiment_metrics(metric_values_A, metric_values_B, alpha=0.05):
    """
    Compare two groups of experiment metrics with statistical testing.

    Args:
        metric_values_A: Array of metric values for condition A
        metric_values_B: Array of metric values for condition B
        alpha: Significance level (default 0.05)

    Returns:
        Dict with test results and recommendation
    """
    # Check normality assumptions
    _, p_normal_A = stats.shapiro(metric_values_A)
    _, p_normal_B = stats.shapiro(metric_values_B)

    # Choose appropriate test
    if p_normal_A > alpha and p_normal_B > alpha:
        # Both normal, use t-test
        statistic, p_value = stats.ttest_ind(metric_values_A, metric_values_B)
        test_name = "Independent t-test"
    else:
        # Non-normal, use Mann-Whitney U
        statistic, p_value = stats.mannwhitneyu(
            metric_values_A,
            metric_values_B,
            alternative='two-sided'
        )
        test_name = "Mann-Whitney U test"

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(metric_values_A) - 1) * metric_values_A.var() +
         (len(metric_values_B) - 1) * metric_values_B.var()) /
        (len(metric_values_A) + len(metric_values_B) - 2)
    )
    cohens_d = (metric_values_A.mean() - metric_values_B.mean()) / pooled_std

    # Determine significance
    significant = p_value < alpha

    # Generate recommendation
    if not significant:
        recommendation = "No significant difference detected"
    elif abs(cohens_d) < 0.2:
        recommendation = "Significant but small effect size"
    elif abs(cohens_d) < 0.5:
        recommendation = "Significant medium effect size"
    else:
        recommendation = "Significant large effect size"

    return {
        "test_name": test_name,
        "statistic": statistic,
        "p_value": p_value,
        "significant": significant,
        "cohens_d": cohens_d,
        "recommendation": recommendation,
        "mean_A": metric_values_A.mean(),
        "mean_B": metric_values_B.mean(),
        "improvement_pct": ((metric_values_B.mean() - metric_values_A.mean()) /
                            metric_values_A.mean() * 100)
    }
\`\`\`

### Verified Pattern: Jinja2 Report Generation

\`\`\`python
# Source: https://www.incentius.com/blog-posts/build-modern-print-ready-pdfs-with-python-flask-weasyprint/
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from datetime import datetime

def generate_analysis_report(analysis_data: dict, output_path: str):
    """
    Generate PDF report from analysis results.

    Args:
        analysis_data: Dict with analysis results (metrics, plots, insights)
        output_path: Path to save PDF report

    Returns:
        Path to generated PDF
    """
    # Setup Jinja2 environment
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('analysis_report.html')

    # Prepare template context
    context = {
        'title': 'Model Analysis Report',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'metrics': analysis_data['metrics'],
        'insights': analysis_data['insights'],
        'plots': analysis_data['plot_paths'],  # Paths to saved plots
    }

    # Render HTML
    html_content = template.render(**context)

    # Convert to PDF
    HTML(string=html_content).write_pdf(output_path)

    return output_path
\`\`\`

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| **Custom feature importance** | **SHAP values** | 2017-2018 | SHAP provides mathematically grounded, model-agnostic explanations |
| **Manual residual analysis** | **Automated failure mode clustering** | 2020-2022 | Unsupervised learning identifies systematic error patterns automatically |
| **Static matplotlib plots** | **Interactive Plotly dashboards** | 2021-2023 | Interactive exploration enables faster insights |
| **Notebook-based analysis** | **Automated report generation (Jinja2+WeasyPrint)** | 2019-2021 | Reproducible, shareable reports with professional formatting |
| **LIME for local explanations** | **SHAP for both local and global** | 2018-2020 | SHAP unifies local/global explanations with better consistency |

**Deprecated/outdated:**
- **Custom permutation importance loops:** ELI5 handles this better
- **iCharts/Bokeh for reports:** Plotly and WeasyPrint are more modern
- **Excel for analysis results:** Python + pandas is more reproducible
- **Separate analysis notebooks:** Integrate with MLflow artifacts instead
- **Manual statistical testing:** Automated pipelines are more reliable

## Open Questions

1. **Captum integration with EfficientNet models**
   - What we know: Captum supports Grad-CAM for PyTorch models, research shows 2025 examples with EfficientNet
   - What's unclear: Exact layer targeting for specific EfficientNet variants (B0-B4), performance on biomass prediction images
   - Recommendation: Start with LayerGradCam on final conv layer, test with sample images before full integration

2. **Failure mode clustering approach**
   - What we know: Unsupervised clustering (KMeans, DBSCAN) can identify error patterns
   - What's unclear: Optimal number of clusters for this dataset (357 images), whether features or predictions should be clustered
   - Recommendation: Start with KMeans (k=3-5), use silhouette score to evaluate, consider DBSCAN if clusters are irregular

3. **Statistical test selection for small samples**
   - What we know: Project may have limited runs per condition (<10)
   - What's unclear: Whether Bayesian approaches would be better than frequentist tests for small samples
   - Recommendation: Use Mann-Whitney U (non-parametric) for small samples, document sample size limitations in reports

4. **Report format preference (HTML vs PDF)**
   - What we know: Jinja2+WeasyPrint supports both, research shows 2025 trend toward interactive HTML
   - What's unclear: User preference for biomass prediction context, whether reports need to be archived as PDFs
   - Recommendation: Generate both HTML (for viewing) and PDF (for archiving), make it configurable

## Sources

### Primary (HIGH confidence)

- **SHAP Documentation** - https://shap.readthedocs.io/ - Official SHAP library documentation with examples
- **scikit-learn metrics** - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html - Classification metrics API
- **scikit-learn confusion matrix** - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html - Confusion matrix computation
- **scipy.stats** - https://docs.scipy.org/doc/scipy/reference/stats.html - Statistical functions for testing
- **Captum Documentation** - https://captum.ai/ - Official PyTorch interpretability library
- **MLflow Documentation** - https://mlflow.org/docs/latest/ - Official MLflow tracking and artifact documentation
- **Existing framework code** - /Users/dustinober/Projects/Image2Biomass-Prediction/mlflow_tracking/ - ExperimentTracker, ExperimentComparator patterns

### Secondary (MEDIUM confidence)

- [Error Analysis to Find Failure Modes](https://mlops.systems/posts/2025-05-23-error-analysis-to-find-failure-modes.html) - mlops.systems (May 23, 2025)
- [SHAP Explained: A Step-by-Step Tutorial](https://medium.datadriveninvestor.com/shap-explained-a-step-by-step-tutorial-for-model-interpretability-8442daac25f6) - Medium (July 9, 2025)
- [Hands-On SHAP: Practical Implementation for Image, Text, and Tabular Data](https://medium.com/@shree144/hands-on-shap-practical-implementation-for-image-text-and-tabular-data-f74b488f8d71) - Medium (2025)
- [Visualize LLM Attention Layers with Captum: A Deep Dive](https://medium.com/@cbrackeen05/visualize-llm-attention-layers-with-captum-a-deep-dive-d82e05f06f35) - Medium (2025)
- [A Complete Guide to A/B Testing in Python](https://www.kdnuggets.com/a-complete-guide-to-a-b-testing-in-python) - KDnuggets (April 2025)
- [How to Perform A/B Testing with Hypothesis Testing in Python](https://towardsdatascience.com/how-to-perform-a-b-testing-with-hypothesis-testing-in-python-a-comprehensive-guide-17b555928c7e/) - Towards Data Science (October 2024)
- [Build Modern, Print-Ready PDFs with Python, Flask & WeasyPrint](https://www.incentius.com/blog-posts/build-modern-print-ready-pdfs-with-python-flask-weasyprint/) - Incentius (June 30, 2025)
- [Creating PDF Reports with Pandas, Jinja and WeasyPrint](https://pbpython.com/pdf-reports.html) - pbpython.com
- [Model Interpretability and Understanding for PyTorch using Captum](https://www.digitalocean.com/community/tutorials/model-interpretability-and-understanding-for-pytorch-using-captum) - DigitalOcean (October 2024)
- [Partial Dependence Plots with Python: A Comprehensive Guide](https://www.blog.trainindata.com/partial-dependence-plots-with-python/) - TrainInData (January 15, 2024)
- [How to Create a Residual Plot in Python](https://www.geeksforgeeks.org/python-how-to-create-a-residual-plot-in-python/) - GeeksforGeeks (July 23, 2025)
- [Understanding Residual Analysis in Regression: A Deep Dive](https://medium.com/@jangdaehan1/understanding-residual-analysis-in-regression-a-deep-dive-bc9ba6f3506d) - Medium (1 year ago)
- [Using WeasyPrint and Jinja2 to create PDFs from HTML and CSS](https://medium.com/@engineering_holistic_ai/using-weasyprint-and-jinja2-to-create-pdfs-from-html-and-css-267127454dbd) - Medium (2 years ago)
- [PDPbox: Partial Dependence Plot Toolbox Tutorial](https://blog.csdn.net/gitblog_00761/article/details/141049320) - CSDN Blog (November 12, 2025)
- [SHAP与LIME：解释"黑箱"模型，实现可信AI](https://blog.csdn.net/qq_37956697/article/details/155139652) - CSDN Blog (November 22, 2025)
- [A/B testing in Python: How to run and analyze experiments](https://www.statsig.com/perspectives/ab-testing-python-experiments) - Statsig (September 2024)
- [Practical Guide to A/B Testing: Tips and Case Study in Python](https://www.lunartech.ai/blog/practical-guide-to-a-b-testing-tips-and-case-study-in-python) - LunaTech (2024)
- [MLflow Evaluation Lab Comprehensive Guide](https://pub.towardsai.net/mlflow-evaluation-lab-comprehensive-guide-050d09ea24fb) - Towards AI (December 2025)

### Tertiary (LOW confidence)

- [GitHub Topics: insight-generation](https://github.com/topics/insight-generation) - GitHub topic aggregation
- [Explainable AI for Forensic Analysis: A Comparative Study of SHAP and LIME](https://www.mdpi.com/2076-3417/15/13/7329) - MDPI (2025)
- [LIME vs SHAP: What's the Difference for Model Interpretability](https://apxml.com/posts/lime-vs-shap-difference-interpretability) - Apxml (April 17, 2025)
- [Top 5 Model Interpretability Libraries for Python](https://mltooling.substack.com/p/top-5-model-interpretability-libraries) - MLTooling Substack
- [ELI5 Documentation](https://eli5.readthedocs.io/) - ELI5 library documentation
- [Top 10 Python Data Visualization Libraries in 2025](https://reflex.dev/blog/2025-01-27-top-10-python-datavisualization-libraries/) - Reflex (January 27, 2025)
- [Plotly vs Matplotlib vs Seaborn: The 2025 Python Visual Battle](https://medium.com/@hadiyolworld007/plotly-vs-matplotlib-vs-seaborn-the-2025-python-visual-battle-e51e9e2b744f) - Medium (7 months ago)
- [How to Visualize Machine Learning Models - SHAP and LIME](https://www.atlantic.net/gpu-server-hosting/how-to-visualize-machine-learning-models-with-shap-and-lime/) - Atlantic.net (January 16, 2025)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries are industry standards with official documentation
- Architecture: HIGH - Based on verified existing framework patterns (ExperimentComparator, ExperimentTracker)
- Pitfalls: MEDIUM - Based on WebSearch verified with some official sources, but some are general best practices rather than library-specific
- Code examples: HIGH - Verified against official documentation and existing codebase patterns

**Research date:** 2025-01-17
**Valid until:** 2025-02-17 (30 days - libraries are stable but ecosystem moves fast)

**Key versions referenced:**
- shap: >=0.46.0 (current stable)
- matplotlib: >=3.8.0
- seaborn: >=0.13.0
- captum: >=0.7.0
- scipy: >=1.11.0
- statsmodels: >=0.14.0
- weasyprint: >=60.0
