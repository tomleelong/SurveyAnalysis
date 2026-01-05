"""HTML report generator for survey analysis."""

import logging
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape

from .analyzer import CrossTabResult, QuestionStats, SurveyAnalysis, SurveyAnalyzer
from .insights import InsightsGenerator
from .models import InsightsConfig, QuestionType, Survey
from .visualizer import InsightsVisualizer, SurveyVisualizer

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate HTML reports from survey analysis."""

    def __init__(
        self,
        survey: Survey,
        analysis: SurveyAnalysis | None = None,
        template_dir: str | Path | None = None,
        insights_config: InsightsConfig | None = None,
    ):
        """Initialize report generator.

        Args:
            survey: Parsed Survey object.
            analysis: Pre-computed analysis (optional, will compute if not provided).
            template_dir: Custom template directory (optional).
            insights_config: Configuration for insights generation (optional).
        """
        self.survey = survey
        self.analysis = analysis or SurveyAnalyzer(survey).analyze()
        self.visualizer = SurveyVisualizer(self.analysis)
        self.analyzer = SurveyAnalyzer(survey)
        self.insights_config = insights_config

        # Set up Jinja2 environment
        if template_dir:
            self.env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=select_autoescape(["html", "xml"]),
            )
        else:
            # Try package templates first, fall back to local
            try:
                self.env = Environment(
                    loader=PackageLoader("survey_analyzer", "../templates"),
                    autoescape=select_autoescape(["html", "xml"]),
                )
            except Exception:
                # Fall back to file system loader
                templates_path = Path(__file__).parent.parent / "templates"
                self.env = Environment(
                    loader=FileSystemLoader(templates_path),
                    autoescape=select_autoescape(["html", "xml"]),
                )

    def generate_report(
        self,
        output_path: str | Path | None = None,
        include_charts: bool = True,
        include_insights: bool = True,
        chart_type: str = "bar",  # 'bar' or 'pie'
        crosstab_pairs: list[tuple[str, str]] | None = None,
    ) -> str:
        """Generate HTML report.

        Args:
            output_path: Path to save the report (optional).
            include_charts: Whether to include interactive charts.
            include_insights: Whether to include advanced insights section.
            chart_type: Default chart type for questions ('bar' or 'pie').
            crosstab_pairs: List of (question_id, question_id) pairs for cross-tabulation.

        Returns:
            HTML string of the report.
        """
        # Generate insights
        key_insights = None
        segmentation_chart = None
        distribution_chart = None
        completion_chart = None
        question_type_chart = None

        if include_insights:
            insights_gen = InsightsGenerator(self.survey, config=self.insights_config)
            insights = insights_gen.generate()
            insights_viz = InsightsVisualizer(insights)

            key_insights = insights.key_insights

            if include_charts:
                segmentation_chart = insights_viz.fig_to_html(
                    insights_viz.create_segmentation_chart()
                )
                distribution_chart = insights_viz.fig_to_html(
                    insights_viz.create_distribution_chart()
                )
                completion_chart = insights_viz.fig_to_html(
                    insights_viz.create_completion_gauge()
                )
                question_type_chart = insights_viz.fig_to_html(
                    insights_viz.create_question_type_chart()
                )

        # Prepare question data with charts
        question_data = []
        for stats in self.analysis.question_stats:
            item = {"stats": stats, "chart": None}

            if include_charts and stats.option_counts:
                # Choose chart type based on question type and data
                if stats.question_type == QuestionType.SINGLE_SELECT and len(stats.option_counts) <= 6:
                    fig = self.visualizer.create_pie_chart(stats)
                else:
                    fig = self.visualizer.create_bar_chart(stats)

                item["chart"] = self.visualizer.fig_to_html(fig)

            question_data.append(item)

        # Generate summary charts
        summary_chart = None
        response_rate_chart = None
        if include_charts:
            summary_chart = self.visualizer.fig_to_html(
                self.visualizer.create_summary_dashboard()
            )
            response_rate_chart = self.visualizer.fig_to_html(
                self.visualizer.create_response_rate_chart()
            )

        # Generate cross-tabulations
        crosstabs = []
        if crosstab_pairs:
            for q1_id, q2_id in crosstab_pairs:
                result = self.analyzer.cross_tabulate(q1_id, q2_id)
                if result:
                    crosstab_data = {
                        "question1_text": result.question1_text,
                        "question2_text": result.question2_text,
                        "chart": None,
                    }
                    if include_charts:
                        fig = self.visualizer.create_stacked_bar_chart(result)
                        crosstab_data["chart"] = self.visualizer.fig_to_html(fig)
                    crosstabs.append(crosstab_data)

        # Render template
        template = self.env.get_template("report.html")
        html = template.render(
            title=self.analysis.survey_title,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            total_responses=self.analysis.total_responses,
            completion_rate=self.analysis.completion_rate,
            avg_completion_time=self.analysis.avg_completion_time_minutes,
            question_count=len(self.analysis.question_stats),
            key_insights=key_insights,
            segmentation_chart=segmentation_chart,
            distribution_chart=distribution_chart,
            completion_chart=completion_chart,
            question_type_chart=question_type_chart,
            summary_chart=summary_chart,
            response_rate_chart=response_rate_chart,
            question_data=question_data,
            crosstabs=crosstabs,
        )

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html, encoding="utf-8")
            logger.info(f"Report saved to {output_path}")

        return html

    def generate_static_report(
        self,
        output_path: str | Path,
        image_format: str = "png",
    ) -> str:
        """Generate report with static images instead of interactive charts.

        Args:
            output_path: Path to save the report.
            image_format: Image format for charts.

        Returns:
            HTML string of the report.
        """
        # This would embed base64 images instead of plotly.js
        # For simplicity, we'll use the same interactive report
        return self.generate_report(output_path=output_path, include_charts=True)


def generate_report(
    survey: Survey,
    output_path: str | Path | None = None,
    insights_config: InsightsConfig | None = None,
    **kwargs,
) -> str:
    """Convenience function to generate a report.

    Args:
        survey: Parsed Survey object.
        output_path: Path to save the report.
        insights_config: Configuration for insights generation (optional).
        **kwargs: Additional arguments for ReportGenerator.generate_report.

    Returns:
        HTML string of the report.
    """
    generator = ReportGenerator(survey, insights_config=insights_config)
    return generator.generate_report(output_path=output_path, **kwargs)
