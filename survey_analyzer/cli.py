"""Command-line interface for Survey Analyzer."""

import json
import logging
import sys
from pathlib import Path

import click

from . import __version__
from .analyzer import SurveyAnalyzer
from .models import InsightsConfig
from .parser import parse_survey
from .reporter import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
def cli():
    """Survey Analyzer - Analyze and visualize Survey Monkey CSV exports."""
    pass


@cli.command()
@click.argument("csv_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path for the report. Defaults to <input_name>_report.html",
)
@click.option(
    "--output-dir",
    "-d",
    type=click.Path(path_type=Path),
    default=Path("output"),
    help="Output directory for reports. Defaults to 'output/'",
)
@click.option(
    "--no-charts",
    is_flag=True,
    help="Generate report without interactive charts.",
)
@click.option(
    "--no-insights",
    is_flag=True,
    help="Generate report without insights section.",
)
@click.option(
    "--segment",
    "-s",
    help="Question ID or text to segment responses by (e.g., Q1 or 'department').",
)
@click.option(
    "--crosstab",
    "-x",
    multiple=True,
    nargs=2,
    help="Cross-tabulate two questions (e.g., -x Q1 Q2). Can be used multiple times.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output.",
)
def analyze(
    csv_file: Path,
    output: Path | None,
    output_dir: Path,
    no_charts: bool,
    no_insights: bool,
    segment: str | None,
    crosstab: tuple,
    verbose: bool,
):
    """Analyze a Survey Monkey CSV export and generate an HTML report.

    CSV_FILE: Path to the Survey Monkey CSV export file.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    click.echo(f"Parsing survey data from {csv_file}...")

    try:
        survey = parse_survey(csv_file)
        click.echo(f"  Found {survey.response_count} responses and {survey.question_count} questions")

        # Determine output path
        if output:
            output_path = output
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{csv_file.stem}_report.html"

        # Build insights config if segmentation question specified
        config = None
        if segment and not no_insights:
            from .models import QuestionMapping
            config = InsightsConfig(
                segmentation_question=QuestionMapping(
                    question_id=segment if segment.startswith("Q") else None,
                    text_pattern=segment if not segment.startswith("Q") else None,
                )
            )

        click.echo(f"Analyzing survey data...")
        generator = ReportGenerator(survey, insights_config=config)

        # Parse cross-tabulation pairs
        crosstab_pairs = list(crosstab) if crosstab else None

        click.echo(f"Generating report...")
        generator.generate_report(
            output_path=output_path,
            include_charts=not no_charts,
            include_insights=not no_insights,
            crosstab_pairs=crosstab_pairs,
        )

        click.echo(f"Report saved to: {output_path}")
        click.echo(click.style("Done!", fg="green", bold=True))

    except Exception as e:
        logger.exception("Error analyzing survey")
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command()
@click.argument("csv_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--question",
    "-q",
    help="Analyze a specific question (by ID like Q1 or by partial text match).",
)
def summary(csv_file: Path, question: str | None):
    """Display summary statistics for a survey.

    CSV_FILE: Path to the Survey Monkey CSV export file.
    """
    click.echo(f"Parsing survey data from {csv_file}...")

    try:
        survey = parse_survey(csv_file)
        analyzer = SurveyAnalyzer(survey)
        analysis = analyzer.analyze()

        click.echo()
        click.echo(click.style(f"Survey: {analysis.survey_title}", bold=True))
        click.echo(f"Total Responses: {analysis.total_responses}")
        click.echo(f"Completion Rate: {analysis.completion_rate:.1f}%")
        if analysis.avg_completion_time_minutes:
            click.echo(f"Avg Completion Time: {analysis.avg_completion_time_minutes:.1f} minutes")
        click.echo()

        if question:
            # Find specific question
            q = survey.get_question_by_id(question) or survey.get_question_by_text(question)
            if not q:
                click.echo(click.style(f"Question not found: {question}", fg="red"))
                return

            stats = analyzer.analyze_question(q)
            _print_question_stats(stats)
        else:
            # Print all questions summary
            click.echo(click.style("Questions:", bold=True))
            for i, stats in enumerate(analysis.question_stats, 1):
                click.echo(f"\n{i}. [{stats.question_id}] {stats.question_text[:60]}...")
                click.echo(f"   Type: {stats.question_type.value} | Responses: {stats.response_count} ({stats.response_rate:.1f}%)")
                if stats.option_counts:
                    top_options = sorted(stats.option_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                    for opt, count in top_options:
                        pct = stats.option_percentages.get(opt, 0)
                        click.echo(f"   - {opt[:40]}: {count} ({pct:.1f}%)")

    except Exception as e:
        logger.exception("Error analyzing survey")
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command()
@click.argument("csv_file", type=click.Path(exists=True, path_type=Path))
@click.option("--q1", required=True, help="First question ID or text.")
@click.option("--q2", required=True, help="Second question ID or text.")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path for the chart.",
)
def crosstab(csv_file: Path, q1: str, q2: str, output: Path | None):
    """Cross-tabulate two questions from a survey.

    CSV_FILE: Path to the Survey Monkey CSV export file.
    """
    click.echo(f"Parsing survey data from {csv_file}...")

    try:
        survey = parse_survey(csv_file)
        analyzer = SurveyAnalyzer(survey)

        # Find questions
        question1 = survey.get_question_by_id(q1) or survey.get_question_by_text(q1)
        question2 = survey.get_question_by_id(q2) or survey.get_question_by_text(q2)

        if not question1:
            click.echo(click.style(f"Question not found: {q1}", fg="red"))
            return
        if not question2:
            click.echo(click.style(f"Question not found: {q2}", fg="red"))
            return

        click.echo(f"Cross-tabulating:")
        click.echo(f"  Q1: {question1.text[:60]}...")
        click.echo(f"  Q2: {question2.text[:60]}...")

        result = analyzer.cross_tabulate(question1.id, question2.id)

        if not result:
            click.echo(click.style("No data for cross-tabulation.", fg="yellow"))
            return

        click.echo()
        click.echo(click.style("Contingency Table:", bold=True))
        click.echo(result.contingency_table.to_string())

        click.echo()
        click.echo(click.style("Row Percentages:", bold=True))
        click.echo(result.percentages_by_row.round(1).to_string())

        if output:
            from .visualizer import SurveyVisualizer
            from .analyzer import SurveyAnalysis

            # Create a minimal analysis for the visualizer
            analysis = SurveyAnalysis(
                survey_title=survey.title,
                total_responses=survey.response_count,
                question_stats=[],
                completion_rate=0,
                avg_completion_time_minutes=None,
            )
            viz = SurveyVisualizer(analysis)
            fig = viz.create_stacked_bar_chart(result)

            if output.suffix == ".html":
                fig.write_html(str(output))
            else:
                fig.write_image(str(output), width=1200, height=600, scale=2)

            click.echo(f"Chart saved to: {output}")

    except Exception as e:
        logger.exception("Error in cross-tabulation")
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command()
@click.argument("csv_file", type=click.Path(exists=True, path_type=Path))
def questions(csv_file: Path):
    """List all questions in a survey with their IDs.

    CSV_FILE: Path to the Survey Monkey CSV export file.
    """
    try:
        survey = parse_survey(csv_file)

        click.echo(click.style(f"\nSurvey: {survey.title}", bold=True))
        click.echo(f"Total Questions: {survey.question_count}\n")

        for q in survey.questions:
            click.echo(f"[{q.id}] ({q.question_type.value})")
            click.echo(f"    {q.text}")
            if q.options:
                click.echo(f"    Options: {len(q.options)}")
            click.echo()

    except Exception as e:
        logger.exception("Error listing questions")
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


def _print_question_stats(stats):
    """Print detailed stats for a question."""
    click.echo(click.style(f"\n{stats.question_text}", bold=True))
    click.echo(f"Type: {stats.question_type.value}")
    click.echo(f"Responses: {stats.response_count} / {stats.total_responses} ({stats.response_rate:.1f}%)")

    if stats.option_counts:
        click.echo("\nOption Breakdown:")
        sorted_options = sorted(stats.option_counts.items(), key=lambda x: x[1], reverse=True)
        for opt, count in sorted_options:
            pct = stats.option_percentages.get(opt, 0)
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            click.echo(f"  {bar} {count:3d} ({pct:5.1f}%) {opt}")

    if stats.mean is not None:
        click.echo(f"\nNumeric Stats:")
        click.echo(f"  Mean: {stats.mean:.2f}")
        click.echo(f"  Median: {stats.median:.2f}")
        click.echo(f"  Std Dev: {stats.std_dev:.2f}")

    if stats.text_responses:
        click.echo(f"\nText Responses ({len(stats.text_responses)} total):")
        for resp in stats.text_responses[:5]:
            click.echo(f"  - {resp[:80]}...")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
