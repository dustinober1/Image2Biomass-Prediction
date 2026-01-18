"""
ReportGenerator - Generate professional HTML/PDF reports from analytics results.

This module provides the ReportGenerator class for creating comprehensive
HTML and PDF reports from error analysis, interpretability, and insights
results using Jinja2 templates.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import base64
import os

# Jinja2 imports
try:
    from jinja2 import Environment, FileSystemLoader, TemplateNotFound
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

# WeasyPrint imports (optional, for PDF generation)
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

# Pandas imports
import pandas as pd


class ReportGenerator:
    """
    Generate professional HTML/PDF reports from analytics results.

    This class provides methods for creating comprehensive reports from
    error analysis, model interpretability, and automated insights results.
    Reports are generated using Jinja2 templates and can be exported as
    HTML or PDF.

    Attributes:
        env: Jinja2 Environment for template loading
        template_dir: Directory containing Jinja2 templates

    Example:
        >>> generator = ReportGenerator()
        >>> report_path = generator.generate_error_analysis_report(
        ...     run_id="abc123",
        ...     error_stats=stats,
        ...     plot_paths=plots,
        ...     failure_modes=failure_modes
        ... )
    """

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize Jinja2 environment for report generation.

        Args:
            template_dir: Optional custom template directory.
                         If None, uses default: mlflow_tracking/analytics/templates

        Raises:
            ImportError: If Jinja2 is not installed
        """
        if not JINJA2_AVAILABLE:
            raise ImportError(
                "Jinja2 is required for report generation. "
                "Install with: pip install jinja2"
            )

        if template_dir is None:
            # Use default template directory
            current_dir = Path(__file__).parent
            template_dir = current_dir / "templates"

        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True
        )

        # Add custom filters
        self.env.filters['datetime'] = self._format_datetime
        self.env.filters['percentage'] = self._format_percentage
        self.env.filters['round'] = self._format_round

    def _format_datetime(self, value: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format datetime string."""
        try:
            dt = datetime.fromisoformat(value) if isinstance(value, str) else value
            return dt.strftime(format_str)
        except (ValueError, TypeError):
            return value

    def _format_percentage(self, value: float, decimals: int = 2) -> str:
        """Format as percentage."""
        return f"{value:.{decimals}f}%"

    def _format_round(self, value: float, decimals: int = 2) -> str:
        """Round to specified decimals."""
        return f"{value:.{decimals}f}"

    def _encode_image(self, image_path: str) -> str:
        """
        Encode image as base64 for embedding in HTML.

        Args:
            image_path: Path to image file

        Returns:
            Base64-encoded image string with data URI prefix
        """
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            base64_str = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/png;base64,{base64_str}"
        except (FileNotFoundError, IOError):
            return ""

    def generate_error_analysis_report(
        self,
        run_id: str,
        error_stats: Dict[str, float],
        plot_paths: Dict[str, str],
        failure_modes: Optional[pd.DataFrame] = None,
        output_path: str = "error_analysis_report.html"
    ) -> str:
        """
        Generate error analysis HTML report.

        Args:
            run_id: MLflow run ID
            error_stats: Dictionary with error statistics
                (mean_abs_error, median_abs_error, max_abs_error, std_residual, percentiles)
            plot_paths: Dictionary with paths to plots
                (residuals_plot, error_distribution_plot, prediction_vs_actual_plot)
            failure_modes: Optional DataFrame with failure mode clusters
            output_path: Output path for HTML report

        Returns:
            Path to generated HTML report

        Raises:
            TemplateNotFound: If error_analysis.html template doesn't exist
        """
        # Load template
        template = self.env.get_template("error_analysis.html")

        # Prepare failure modes summary
        failure_modes_summary = None
        if failure_modes is not None:
            failure_modes_summary = []
            for cluster_id in sorted(failure_modes['cluster'].unique()):
                cluster_data = failure_modes[failure_modes['cluster'] == cluster_id]
                failure_modes_summary.append({
                    'cluster_id': int(cluster_id),
                    'sample_count': len(cluster_data),
                    'mean_abs_error': float(cluster_data['abs_residual'].mean()),
                    'mean_pct_error': float(cluster_data['pct_error'].mean())
                })

        # Prepare context
        context = {
            'run_id': run_id,
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'error_stats': error_stats,
            'plot_paths': {
                'residuals_plot': self._encode_image(plot_paths.get('residuals_plot', '')),
                'error_distribution_plot': self._encode_image(plot_paths.get('error_distribution_plot', '')),
                'prediction_vs_actual_plot': self._encode_image(plot_paths.get('prediction_vs_actual_plot', ''))
            },
            'failure_modes': failure_modes_summary
        }

        # Render template
        html_content = template.render(**context)

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)

        return str(output_path)

    def generate_interpretability_report(
        self,
        run_id: str,
        feature_importance: pd.DataFrame,
        shap_summary_path: str,
        local_explanation_path: str,
        permutation_importance: Optional[pd.DataFrame] = None,
        output_path: str = "interpretability_report.html"
    ) -> str:
        """
        Generate interpretability HTML report.

        Args:
            run_id: MLflow run ID
            feature_importance: DataFrame with feature importance (columns: feature, importance)
            shap_summary_path: Path to SHAP summary plot
            local_explanation_path: Path to local explanation plot
            permutation_importance: Optional DataFrame with permutation importance
            output_path: Output path for HTML report

        Returns:
            Path to generated HTML report

        Raises:
            TemplateNotFound: If interpretability_report.html template doesn't exist
        """
        # Load template
        template = self.env.get_template("interpretability_report.html")

        # Prepare feature importance summary
        feature_importance_summary = []
        if feature_importance is not None and not feature_importance.empty:
            for _, row in feature_importance.head(20).iterrows():
                feature_importance_summary.append({
                    'feature': str(row.get('feature', row.index[0] if len(row.index) > 0 else 'Unknown')),
                    'importance': float(row.get('importance', row.get('mean_abs_shap', 0)))
                })

        # Prepare permutation importance summary
        permutation_summary = None
        if permutation_importance is not None and not permutation_importance.empty:
            permutation_summary = []
            for _, row in permutation_importance.head(10).iterrows():
                permutation_summary.append({
                    'feature': str(row.get('feature', 'Unknown')),
                    'importance': float(row.get('importance', 0))
                })

        # Prepare context
        context = {
            'run_id': run_id,
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'feature_importance': feature_importance_summary,
            'shap_summary_plot': self._encode_image(shap_summary_path),
            'local_explanation_plot': self._encode_image(local_explanation_path),
            'permutation_importance': permutation_summary
        }

        # Render template
        html_content = template.render(**context)

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)

        return str(output_path)

    def generate_insights_report(
        self,
        run_ids: List[str],
        insights: Dict[str, Any],
        correlations: Optional[pd.DataFrame] = None,
        rankings: Optional[pd.DataFrame] = None,
        output_path: str = "insights_report.html"
    ) -> str:
        """
        Generate insights summary HTML report.

        Args:
            run_ids: List of MLflow run IDs
            insights: Dictionary with insights (best_run, summary_statistics, statistical_tests, recommendation)
            correlations: Optional DataFrame with hyperparameter correlations
            rankings: Optional DataFrame with experiment rankings
            output_path: Output path for HTML report

        Returns:
            Path to generated HTML report

        Raises:
            TemplateNotFound: If insights_summary.html template doesn't exist
        """
        # Load template
        template = self.env.get_template("insights_summary.html")

        # Prepare statistical tests summary
        statistical_tests_summary = []
        if 'statistical_tests' in insights:
            for test_name, test_result in insights['statistical_tests'].items():
                if isinstance(test_result, dict):
                    statistical_tests_summary.append({
                        'name': test_name,
                        'p_value': float(test_result.get('p_value', 1.0)),
                        'effect_size': float(test_result.get('effect_size', 0.0)),
                        'interpretation': str(test_result.get('interpretation', 'N/A'))
                    })

        # Prepare correlations summary
        correlations_summary = None
        if correlations is not None and not correlations.empty:
            correlations_summary = []
            for _, row in correlations.head(10).iterrows():
                correlations_summary.append({
                    'parameter': str(row.get('parameter', 'Unknown')),
                    'correlation': float(row.get('correlation', 0.0)),
                    'interpretation': str(row.get('interpretation', 'N/A'))
                })

        # Prepare rankings summary
        rankings_summary = None
        if rankings is not None and not rankings.empty:
            rankings_summary = []
            for _, row in rankings.head(10).iterrows():
                rankings_summary.append({
                    'run_id': str(row.get('run_id', 'Unknown')),
                    'score': float(row.get('score', 0.0)),
                    'metrics': dict(row.get('metrics', {}))
                })

        # Prepare context
        context = {
            'run_ids': run_ids,
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'best_run': insights.get('best_run', 'N/A'),
            'summary_statistics': insights.get('summary_statistics', {}),
            'statistical_tests': statistical_tests_summary,
            'recommendation': insights.get('recommendation', ''),
            'correlations': correlations_summary,
            'rankings': rankings_summary
        }

        # Render template
        html_content = template.render(**context)

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)

        return str(output_path)

    def generate_comprehensive_report(
        self,
        run_ids: List[str],
        error_analysis: Optional[Dict] = None,
        interpretability: Optional[Dict] = None,
        insights: Optional[Dict] = None,
        output_format: str = "html",
        output_path: str = "comprehensive_report"
    ) -> str:
        """
        Generate comprehensive report combining all analyses.

        Args:
            run_ids: List of MLflow run IDs
            error_analysis: Optional dictionary with error analysis results
            interpretability: Optional dictionary with interpretability results
            insights: Optional dictionary with insights results
            output_format: Output format (html or pdf)
            output_path: Output path (without extension)

        Returns:
            Path to generated report
        """
        # Create HTML by combining all available analyses
        html_sections = []

        # Add header
        html_sections.append(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Comprehensive Analytics Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .section {{ margin-bottom: 40px; }}
                .metadata {{ background: #ecf0f1; padding: 15px; border-radius: 5px; }}
                .run-id {{ font-family: monospace; background: #f8f9fa; padding: 2px 6px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <h1>Comprehensive Analytics Report</h1>
            <div class="metadata">
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Run IDs:</strong> {', '.join([f'<span class="run-id">{rid}</span>' for rid in run_ids])}</p>
            </div>
        """)

        # Add error analysis section if available
        if error_analysis:
            html_sections.append(f"""
            <div class="section">
                <h2>Error Analysis</h2>
                <p>Error analysis results for run <span class="run-id">{error_analysis.get('run_id', 'N/A')}</span></p>
            </div>
            """)

        # Add interpretability section if available
        if interpretability:
            html_sections.append(f"""
            <div class="section">
                <h2>Model Interpretability</h2>
                <p>Interpretability analysis for run <span class="run-id">{interpretability.get('run_id', 'N/A')}</span></p>
            </div>
            """)

        # Add insights section if available
        if insights:
            html_sections.append(f"""
            <div class="section">
                <h2>Automated Insights</h2>
                <p>Insights generated from {len(run_ids)} runs</p>
            </div>
            """)

        # Add footer
        html_sections.append("""
        </body>
        </html>
        """)

        # Combine all sections
        html_content = '\n'.join(html_sections)

        # Determine output path
        if output_format == "pdf":
            output_file = f"{output_path}.pdf"
            # Convert HTML to PDF
            return self.convert_html_to_pdf_content(html_content, output_file)
        else:
            output_file = f"{output_path}.html"
            # Save HTML
            output_path_obj = Path(output_file)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path_obj, 'w') as f:
                f.write(html_content)
            return str(output_path_obj)

    def convert_html_to_pdf(self, html_path: str, pdf_path: str) -> str:
        """
        Convert HTML report to PDF using WeasyPrint.

        Args:
            html_path: Path to HTML file
            pdf_path: Path to output PDF file

        Returns:
            Path to generated PDF

        Raises:
            ImportError: If WeasyPrint is not installed
        """
        if not WEASYPRINT_AVAILABLE:
            raise ImportError(
                "WeasyPrint is required for PDF generation. "
                "Install with: pip install weasyprint"
            )

        # Read HTML content
        with open(html_path, 'r') as f:
            html_content = f.read()

        return self.convert_html_to_pdf_content(html_content, pdf_path)

    def convert_html_to_pdf_content(self, html_content: str, pdf_path: str) -> str:
        """
        Convert HTML content to PDF using WeasyPrint.

        Args:
            html_content: HTML content as string
            pdf_path: Path to output PDF file

        Returns:
            Path to generated PDF

        Raises:
            ImportError: If WeasyPrint is not installed
        """
        if not WEASYPRINT_AVAILABLE:
            raise ImportError(
                "WeasyPrint is required for PDF generation. "
                "Install with: pip install weasyprint"
            )

        # Create output directory if needed
        pdf_path_obj = Path(pdf_path)
        pdf_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Convert to PDF
        HTML(string=html_content).write_pdf(pdf_path_obj)

        return str(pdf_path_obj)


def generate_html_report(
    report_type: str,
    **kwargs
) -> str:
    """
    Convenience function to generate HTML reports.

    Args:
        report_type: Type of report (error_analysis, interpretability, insights)
        **kwargs: Arguments to pass to specific report generator

    Returns:
        Path to generated HTML report
    """
    generator = ReportGenerator()

    if report_type == "error_analysis":
        return generator.generate_error_analysis_report(**kwargs)
    elif report_type == "interpretability":
        return generator.generate_interpretability_report(**kwargs)
    elif report_type == "insights":
        return generator.generate_insights_report(**kwargs)
    else:
        raise ValueError(f"Unknown report type: {report_type}")


def generate_pdf_report(
    html_path: str,
    pdf_path: str
) -> str:
    """
    Convenience function to convert HTML to PDF.

    Args:
        html_path: Path to HTML file
        pdf_path: Path to output PDF file

    Returns:
        Path to generated PDF
    """
    generator = ReportGenerator()
    return generator.convert_html_to_pdf(html_path, pdf_path)
