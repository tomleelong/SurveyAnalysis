"""Plotly-based visualization for survey data."""

import base64
from io import BytesIO
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analyzer import CrossTabResult, QuestionStats, SurveyAnalysis
from .models import QuestionType


# Bertram Brand Color Palette
BERTRAM_COLORS = {
    "white": "#FFFFFF",
    "cream": "#F2EFE9",
    "blue": "#225AA8",
    "navy": "#232C40",
    "royal": "#1C2334",
}

# Color palette for charts - ordered for visual hierarchy
COLORS = [
    BERTRAM_COLORS["blue"],    # Primary - Bertram Blue
    BERTRAM_COLORS["navy"],    # Secondary - Bertram Navy
    BERTRAM_COLORS["royal"],   # Tertiary - Bertram Royal
    "#4A7DC4",                 # Lighter blue (derived)
    "#3D4A5C",                 # Lighter navy (derived)
    "#2E3A4D",                 # Mid tone (derived)
    "#6B8FD4",                 # Even lighter blue
    "#5C6B7A",                 # Light slate
]

CHART_TEMPLATE = "plotly_white"


class SurveyVisualizer:
    """Generate visualizations for survey analysis results."""

    def __init__(self, analysis: SurveyAnalysis):
        """Initialize visualizer with analysis results.

        Args:
            analysis: SurveyAnalysis from the analyzer.
        """
        self.analysis = analysis

    def create_bar_chart(
        self,
        stats: QuestionStats,
        horizontal: bool = True,
        show_percentages: bool = True,
    ) -> go.Figure:
        """Create a bar chart for a question's responses.

        Args:
            stats: QuestionStats for the question.
            horizontal: If True, create horizontal bars.
            show_percentages: If True, show percentages on bars.

        Returns:
            Plotly Figure object.
        """
        if not stats.option_counts:
            return self._create_empty_chart(stats.question_text)

        # Sort by count descending
        sorted_items = sorted(
            stats.option_counts.items(), key=lambda x: x[1], reverse=True
        )
        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        percentages = [stats.option_percentages.get(label, 0) for label in labels]

        # Truncate long labels
        display_labels = [
            label[:40] + "..." if len(label) > 40 else label for label in labels
        ]

        if horizontal:
            fig = go.Figure(
                go.Bar(
                    y=display_labels[::-1],  # Reverse for top-to-bottom reading
                    x=values[::-1],
                    orientation="h",
                    marker_color=COLORS[0],
                    text=[f"{v} ({p:.1f}%)" for v, p in zip(values[::-1], percentages[::-1])]
                    if show_percentages
                    else values[::-1],
                    textposition="auto",
                )
            )
        else:
            fig = go.Figure(
                go.Bar(
                    x=display_labels,
                    y=values,
                    marker_color=COLORS[0],
                    text=[f"{v} ({p:.1f}%)" for v, p in zip(values, percentages)]
                    if show_percentages
                    else values,
                    textposition="auto",
                )
            )

        fig.update_layout(
            title=dict(
                text=self._truncate_title(stats.question_text),
                font=dict(size=14),
            ),
            template=CHART_TEMPLATE,
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20),
            height=max(300, len(labels) * 30) if horizontal else 400,
        )

        return fig

    def create_pie_chart(self, stats: QuestionStats, hole: float = 0.3) -> go.Figure:
        """Create a pie/donut chart for a question's responses.

        Args:
            stats: QuestionStats for the question.
            hole: Size of hole for donut chart (0 for pie).

        Returns:
            Plotly Figure object.
        """
        if not stats.option_counts:
            return self._create_empty_chart(stats.question_text)

        # Sort by count and limit to top 10 for readability
        sorted_items = sorted(
            stats.option_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]
        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        # Truncate long labels
        display_labels = [
            label[:30] + "..." if len(label) > 30 else label for label in labels
        ]

        fig = go.Figure(
            go.Pie(
                labels=display_labels,
                values=values,
                hole=hole,
                marker=dict(colors=COLORS[: len(labels)]),
                textinfo="percent+label",
                textposition="outside",
            )
        )

        fig.update_layout(
            title=dict(
                text=self._truncate_title(stats.question_text),
                font=dict(size=14),
            ),
            template=CHART_TEMPLATE,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            margin=dict(l=20, r=20, t=60, b=80),
            height=450,
        )

        return fig

    def create_stacked_bar_chart(self, crosstab: CrossTabResult) -> go.Figure:
        """Create a stacked bar chart for cross-tabulation.

        Args:
            crosstab: CrossTabResult from analyzer.

        Returns:
            Plotly Figure object.
        """
        df = crosstab.percentages_by_row

        fig = go.Figure()

        for i, col in enumerate(df.columns):
            fig.add_trace(
                go.Bar(
                    name=str(col)[:30],
                    x=[str(idx)[:30] for idx in df.index],
                    y=df[col],
                    marker_color=COLORS[i % len(COLORS)],
                    text=[f"{v:.1f}%" for v in df[col]],
                    textposition="inside",
                )
            )

        fig.update_layout(
            title=dict(
                text=f"{self._truncate_title(crosstab.question1_text)} by {self._truncate_title(crosstab.question2_text)}",
                font=dict(size=14),
            ),
            barmode="stack",
            template=CHART_TEMPLATE,
            xaxis_title=crosstab.question1_text[:50],
            yaxis_title="Percentage",
            legend=dict(orientation="h", yanchor="bottom", y=-0.4),
            margin=dict(l=20, r=20, t=60, b=100),
            height=500,
        )

        return fig

    def create_heatmap(self, crosstab: CrossTabResult) -> go.Figure:
        """Create a heatmap for cross-tabulation.

        Args:
            crosstab: CrossTabResult from analyzer.

        Returns:
            Plotly Figure object.
        """
        df = crosstab.contingency_table

        # Custom Bertram colorscale from cream to blue
        bertram_colorscale = [
            [0, BERTRAM_COLORS["cream"]],
            [0.5, "#4A7DC4"],
            [1, BERTRAM_COLORS["blue"]],
        ]

        fig = go.Figure(
            go.Heatmap(
                z=df.values,
                x=[str(c)[:25] for c in df.columns],
                y=[str(i)[:25] for i in df.index],
                colorscale=bertram_colorscale,
                text=df.values,
                texttemplate="%{text}",
                textfont=dict(size=12),
                hovertemplate="Row: %{y}<br>Col: %{x}<br>Count: %{z}<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(
                text=f"{self._truncate_title(crosstab.question1_text)} vs {self._truncate_title(crosstab.question2_text)}",
                font=dict(size=14),
            ),
            template=CHART_TEMPLATE,
            xaxis_title=crosstab.question2_text[:40],
            yaxis_title=crosstab.question1_text[:40],
            margin=dict(l=20, r=20, t=60, b=80),
            height=max(350, len(df.index) * 35),
        )

        return fig

    def create_summary_dashboard(self) -> go.Figure:
        """Create a summary dashboard with key metrics.

        Returns:
            Plotly Figure with multiple subplots.
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}],
            ],
            subplot_titles=[
                "Total Responses",
                "Completion Rate",
                "Avg Completion Time",
                "Total Questions",
            ],
        )

        # Total responses
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=self.analysis.total_responses,
                number=dict(font=dict(size=48, color=COLORS[0])),
            ),
            row=1,
            col=1,
        )

        # Completion rate
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=self.analysis.completion_rate,
                number=dict(suffix="%", font=dict(size=36)),
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color=COLORS[1]),
                    bgcolor="lightgray",
                ),
            ),
            row=1,
            col=2,
        )

        # Avg completion time
        avg_time = self.analysis.avg_completion_time_minutes or 0
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=avg_time,
                number=dict(suffix=" min", font=dict(size=48, color=COLORS[2])),
            ),
            row=2,
            col=1,
        )

        # Total questions
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=len(self.analysis.question_stats),
                number=dict(font=dict(size=48, color=COLORS[3])),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title=dict(
                text=f"Survey Summary: {self.analysis.survey_title}",
                font=dict(size=18),
            ),
            template=CHART_TEMPLATE,
            height=500,
            margin=dict(l=20, r=20, t=80, b=20),
        )

        return fig

    def create_response_rate_chart(self) -> go.Figure:
        """Create a chart showing response rates per question.

        Returns:
            Plotly Figure object.
        """
        questions = []
        rates = []

        for stats in self.analysis.question_stats:
            questions.append(self._truncate_title(stats.question_text, 50))
            rates.append(stats.response_rate)

        fig = go.Figure(
            go.Bar(
                y=questions[::-1],
                x=rates[::-1],
                orientation="h",
                marker_color=[
                    COLORS[0] if r >= 80 else COLORS[2] if r >= 50 else COLORS[4]
                    for r in rates[::-1]
                ],
                text=[f"{r:.1f}%" for r in rates[::-1]],
                textposition="auto",
            )
        )

        fig.update_layout(
            title=dict(text="Response Rate by Question", font=dict(size=16)),
            xaxis_title="Response Rate (%)",
            template=CHART_TEMPLATE,
            height=max(400, len(questions) * 25),
            margin=dict(l=20, r=20, t=60, b=40),
        )

        return fig

    def _create_empty_chart(self, title: str) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title=dict(text=self._truncate_title(title), font=dict(size=14)),
            template=CHART_TEMPLATE,
            height=300,
        )
        return fig

    def _truncate_title(self, text: str, max_length: int = 80) -> str:
        """Truncate title text if too long."""
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    @staticmethod
    def fig_to_html(fig: go.Figure, full_html: bool = False) -> str:
        """Convert figure to HTML string.

        Args:
            fig: Plotly Figure.
            full_html: If True, include full HTML document structure.

        Returns:
            HTML string.
        """
        return fig.to_html(full_html=full_html, include_plotlyjs="cdn")

    @staticmethod
    def fig_to_image(
        fig: go.Figure, format: str = "png", width: int = 1200, height: int = 600
    ) -> bytes:
        """Convert figure to image bytes.

        Args:
            fig: Plotly Figure.
            format: Image format (png, svg, pdf, jpeg).
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Image bytes.
        """
        return fig.to_image(format=format, width=width, height=height, scale=2)

    @staticmethod
    def fig_to_base64(
        fig: go.Figure, format: str = "png", width: int = 1200, height: int = 600
    ) -> str:
        """Convert figure to base64 encoded image.

        Args:
            fig: Plotly Figure.
            format: Image format.
            width: Image width.
            height: Image height.

        Returns:
            Base64 encoded string.
        """
        img_bytes = SurveyVisualizer.fig_to_image(fig, format, width, height)
        return base64.b64encode(img_bytes).decode("utf-8")

    def save_chart(
        self,
        fig: go.Figure,
        output_path: str | Path,
        format: str = "png",
        width: int = 1200,
        height: int = 600,
    ) -> None:
        """Save a chart to file.

        Args:
            fig: Plotly Figure.
            output_path: Output file path.
            format: Image format (png, svg, pdf, html).
            width: Image width.
            height: Image height.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "html":
            fig.write_html(str(output_path))
        else:
            fig.write_image(str(output_path), width=width, height=height, scale=2)
