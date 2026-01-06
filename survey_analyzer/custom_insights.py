"""Custom analysis module for survey-specific insights.

This module allows configuring domain-specific analyses that go beyond
the generic insights system. Configure analyses via Python dictionaries.
"""

import logging
from dataclasses import dataclass, field

from .insights import InsightsGenerator
from .models import CustomAnalysisConfig, CustomConfig, Survey
from .visualizer import InsightsVisualizer

logger = logging.getLogger(__name__)


@dataclass
class CustomAnalysisSection:
    """A section of custom analysis with chart and insights."""

    title: str
    description: str = ""
    chart: str | None = None  # HTML chart string
    insights: list[dict] = field(default_factory=list)
    table: dict | None = None  # {"headers": [...], "rows": [[...]]}


@dataclass
class CustomAnalysisResult:
    """Complete custom analysis results."""

    title: str
    sections: list[CustomAnalysisSection] = field(default_factory=list)


class CustomInsightsGenerator:
    """Generate custom, survey-specific insights based on configuration."""

    def __init__(self, survey: Survey, config: dict | CustomConfig):
        """Initialize with survey data and configuration.

        Args:
            survey: The survey data to analyze.
            config: Configuration dict or CustomConfig object.
        """
        self.survey = survey
        self.insights_gen = InsightsGenerator(survey)
        self.visualizer = InsightsVisualizer(self.insights_gen.generate())

        # Parse config
        if isinstance(config, dict):
            self.config = CustomConfig(**config)
        else:
            self.config = config

    def generate(self) -> CustomAnalysisResult:
        """Generate custom analysis based on configuration.

        Returns:
            CustomAnalysisResult with all configured analyses.
        """
        result = CustomAnalysisResult(title=self.config.title)

        for analysis_config in self.config.analyses:
            section = self._process_analysis(analysis_config)
            if section:
                result.sections.append(section)

        return result

    def _process_analysis(
        self, config: CustomAnalysisConfig
    ) -> CustomAnalysisSection | None:
        """Process a single analysis configuration.

        Args:
            config: Configuration for this analysis.

        Returns:
            CustomAnalysisSection or None if analysis failed.
        """
        analysis_type = config.type.lower().replace("-", "_").replace(" ", "_")

        if analysis_type == "cross_question_metric":
            return self._analyze_cross_question_metric(config)
        elif analysis_type == "matrix_sentiment":
            return self._analyze_matrix_sentiment(config)
        elif analysis_type == "response_breakdown":
            return self._analyze_response_breakdown(config)
        elif analysis_type == "numeric_histogram":
            return self._analyze_numeric_histogram(config)
        else:
            logger.warning(f"Unknown analysis type: {config.type}")
            return None

    def _analyze_cross_question_metric(
        self, config: CustomAnalysisConfig
    ) -> CustomAnalysisSection | None:
        """Analyze a metric segmented by another question."""
        if not config.segment_question or not config.metric_question:
            logger.warning("cross_question_metric requires segment_question and metric_question")
            return None

        analysis = self.insights_gen.analyze_cross_question(
            segment_question_text=config.segment_question,
            metric_question_text=config.metric_question,
            metric_values=config.metric_values,
        )

        if not analysis:
            return None

        # Create chart
        chart_html = self.visualizer.fig_to_html(
            self.visualizer.create_cross_question_chart(analysis, title=config.title)
        )

        # Generate insights
        insights = []
        if analysis.segments:
            # Find highest and lowest
            highest = analysis.segments[0]
            lowest = analysis.segments[-1] if len(analysis.segments) > 1 else None

            insights.append({
                "title": f"Top: {highest.segment_name}",
                "description": f"{highest.metric_count}/{highest.total_count} respondents ({highest.metric_rate:.0f}%)",
                "metric": f"{highest.metric_rate:.0f}%",
                "severity": "success",
            })

            if lowest and lowest.metric_rate < highest.metric_rate * 0.5:
                insights.append({
                    "title": f"Gap: {lowest.segment_name}",
                    "description": f"Only {lowest.metric_count}/{lowest.total_count} respondents ({lowest.metric_rate:.0f}%)",
                    "metric": f"{lowest.metric_rate:.0f}%",
                    "severity": "warning",
                })

        return CustomAnalysisSection(
            title=config.title,
            description=config.description,
            chart=chart_html,
            insights=insights,
        )

    def _analyze_matrix_sentiment(
        self, config: CustomAnalysisConfig
    ) -> CustomAnalysisSection | None:
        """Analyze sentiment from a matrix question."""
        if not config.question or not config.sentiment_map:
            logger.warning("matrix_sentiment requires question and sentiment_map")
            return None

        analysis = self.insights_gen.analyze_matrix_sentiment(
            question_text=config.question,
            sentiment_map=config.sentiment_map,
        )

        if not analysis:
            return None

        # Create chart
        chart_html = self.visualizer.fig_to_html(
            self.visualizer.create_matrix_sentiment_chart(analysis, title=config.title)
        )

        # Generate insights
        insights = []
        if analysis.items:
            # Top item
            top = analysis.items[0]
            insights.append({
                "title": f"Highest: {top.item_name}",
                "description": f"Average score: {top.avg_score:.2f} ({top.strong_positive_pct:.0f}% strong attachment)",
                "metric": f"{top.avg_score:.2f}",
                "severity": "success",
            })

            # Count items with negative sentiment
            max_score = max(config.sentiment_map.values())
            min_score = min(config.sentiment_map.values())
            mid_score = (max_score + min_score) / 2

            low_items = [i for i in analysis.items if i.avg_score < mid_score]
            if low_items:
                insights.append({
                    "title": "Below Average",
                    "description": f"{len(low_items)} item(s) have below-average sentiment",
                    "metric": str(len(low_items)),
                    "severity": "info",
                })

        return CustomAnalysisSection(
            title=config.title,
            description=config.description,
            chart=chart_html,
            insights=insights,
        )

    def _analyze_response_breakdown(
        self, config: CustomAnalysisConfig
    ) -> CustomAnalysisSection | None:
        """Analyze response distribution for a question."""
        if not config.question:
            logger.warning("response_breakdown requires question")
            return None

        distribution = self.insights_gen.analyze_response_breakdown(config.question)

        if not distribution:
            return None

        # Create chart
        chart_html = self.visualizer.fig_to_html(
            self.visualizer.create_response_breakdown_chart(
                distribution,
                title=config.title,
                highlight_value=config.highlight_value,
            )
        )

        # Generate insights
        insights = []
        if distribution.top_options:
            top = distribution.top_options[0]
            insights.append({
                "title": f"Top Response",
                "description": f'"{top[0][:30]}..." selected by {top[2]:.0f}% of respondents',
                "metric": f"{top[2]:.0f}%",
                "severity": "info",
            })

            # Check for highlighted value
            if config.highlight_value:
                for opt, count, pct in distribution.top_options:
                    if config.highlight_value.lower() in opt.lower():
                        insights.append({
                            "title": config.highlight_value,
                            "description": f"{count} respondents ({pct:.0f}%) selected this option",
                            "metric": f"{pct:.0f}%",
                            "severity": "success" if pct >= 30 else "info",
                        })
                        break

        return CustomAnalysisSection(
            title=config.title,
            description=config.description,
            chart=chart_html,
            insights=insights,
        )

    def _analyze_numeric_histogram(
        self, config: CustomAnalysisConfig
    ) -> CustomAnalysisSection | None:
        """Analyze numeric distribution with histogram."""
        if not config.question:
            logger.warning("numeric_histogram requires question")
            return None

        distribution = self.insights_gen.analyze_numeric_distribution(config.question)

        if not distribution:
            return None

        # Create chart
        chart_html = self.visualizer.fig_to_html(
            self.visualizer.create_numeric_histogram_chart(
                distribution,
                title=config.title,
                show_stats=config.show_stats,
            )
        )

        # Generate insights
        insights = []
        insights.append({
            "title": "Average Score",
            "description": f"Mean: {distribution.mean:.1f}, Median: {distribution.median:.1f}",
            "metric": f"{distribution.mean:.1f}",
            "severity": "info",
        })

        # Check distribution spread
        if distribution.std_dev > 2:
            insights.append({
                "title": "High Variation",
                "description": f"Standard deviation of {distribution.std_dev:.1f} indicates diverse opinions",
                "metric": f"±{distribution.std_dev:.1f}",
                "severity": "warning",
            })

        # Check for strong positive/negative
        if distribution.mean >= distribution.max_val * 0.7:
            insights.append({
                "title": "Strong Positive",
                "description": f"Average of {distribution.mean:.1f} is in the top 30% of the scale",
                "metric": "High",
                "severity": "success",
            })

        return CustomAnalysisSection(
            title=config.title,
            description=config.description,
            chart=chart_html,
            insights=insights,
        )


def generate_custom_analysis(
    survey: Survey,
    config: dict | CustomConfig,
) -> CustomAnalysisResult:
    """Convenience function to generate custom analysis.

    Args:
        survey: Parsed Survey object.
        config: Configuration dict or CustomConfig object.

    Returns:
        CustomAnalysisResult with all configured analyses.
    """
    generator = CustomInsightsGenerator(survey, config)
    return generator.generate()
