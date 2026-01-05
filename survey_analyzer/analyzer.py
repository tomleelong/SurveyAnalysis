"""Statistical analysis engine for survey data."""

from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .models import Question, QuestionType, Survey


@dataclass
class QuestionStats:
    """Statistics for a single question."""

    question_id: str
    question_text: str
    question_type: QuestionType
    total_responses: int
    response_count: int  # Non-empty responses
    response_rate: float
    option_counts: dict[str, int]
    option_percentages: dict[str, float]
    mean: float | None = None
    median: float | None = None
    std_dev: float | None = None
    other_responses: list[str] | None = None
    text_responses: list[str] | None = None


@dataclass
class CrossTabResult:
    """Result of cross-tabulation analysis."""

    question1_id: str
    question1_text: str
    question2_id: str
    question2_text: str
    contingency_table: pd.DataFrame
    percentages_by_row: pd.DataFrame
    percentages_by_col: pd.DataFrame


@dataclass
class SurveyAnalysis:
    """Complete analysis results for a survey."""

    survey_title: str
    total_responses: int
    question_stats: list[QuestionStats]
    completion_rate: float
    avg_completion_time_minutes: float | None


class SurveyAnalyzer:
    """Analyzer for survey data."""

    def __init__(self, survey: Survey):
        """Initialize analyzer with a survey.

        Args:
            survey: Parsed Survey object.
        """
        self.survey = survey

    def analyze(self) -> SurveyAnalysis:
        """Perform complete analysis of the survey.

        Returns:
            SurveyAnalysis with all statistics.
        """
        question_stats = [
            self.analyze_question(q) for q in self.survey.questions
        ]

        completion_rate = self._calculate_completion_rate()
        avg_time = self._calculate_avg_completion_time()

        return SurveyAnalysis(
            survey_title=self.survey.title,
            total_responses=self.survey.response_count,
            question_stats=question_stats,
            completion_rate=completion_rate,
            avg_completion_time_minutes=avg_time,
        )

    def analyze_question(self, question: Question) -> QuestionStats:
        """Analyze a single question.

        Args:
            question: Question to analyze.

        Returns:
            QuestionStats with counts and percentages.
        """
        total = self.survey.response_count
        option_counts: Counter[str] = Counter()
        numeric_values: list[float] = []
        other_responses: list[str] = []
        text_responses: list[str] = []
        responses_with_answer = 0

        for response in self.survey.responses:
            answer = response.answers.get(question.id)
            if not answer:
                continue

            has_response = False

            # Count selected options
            for opt in answer.selected_options:
                option_counts[opt] += 1
                has_response = True

            # Collect numeric values for scale questions
            if answer.numeric_value is not None:
                numeric_values.append(answer.numeric_value)
                has_response = True

            # Collect other text responses
            if answer.other_text:
                other_responses.append(answer.other_text)
                has_response = True

            # Collect open text responses
            if answer.text_value:
                text_responses.append(answer.text_value)
                has_response = True

            if has_response:
                responses_with_answer += 1

        # Calculate percentages
        option_percentages = {
            opt: (count / total * 100) if total > 0 else 0
            for opt, count in option_counts.items()
        }

        # Calculate numeric stats if applicable
        mean = None
        median = None
        std_dev = None
        if numeric_values:
            series = pd.Series(numeric_values)
            mean = float(series.mean())
            median = float(series.median())
            std_dev = float(series.std()) if len(numeric_values) > 1 else 0.0

        return QuestionStats(
            question_id=question.id,
            question_text=question.text,
            question_type=question.question_type,
            total_responses=total,
            response_count=responses_with_answer,
            response_rate=(responses_with_answer / total * 100) if total > 0 else 0,
            option_counts=dict(option_counts),
            option_percentages=option_percentages,
            mean=mean,
            median=median,
            std_dev=std_dev,
            other_responses=other_responses if other_responses else None,
            text_responses=text_responses if text_responses else None,
        )

    def cross_tabulate(
        self, question1_id: str, question2_id: str
    ) -> CrossTabResult | None:
        """Cross-tabulate two questions.

        Args:
            question1_id: ID of first question (rows).
            question2_id: ID of second question (columns).

        Returns:
            CrossTabResult or None if questions not found.
        """
        q1 = self.survey.get_question_by_id(question1_id)
        q2 = self.survey.get_question_by_id(question2_id)

        if not q1 or not q2:
            return None

        # Build data for cross-tabulation
        rows: list[dict[str, Any]] = []
        for response in self.survey.responses:
            a1 = response.answers.get(question1_id)
            a2 = response.answers.get(question2_id)

            if not a1 or not a2:
                continue

            # Get primary answer for each question
            val1 = self._get_primary_answer(a1, q1)
            val2 = self._get_primary_answer(a2, q2)

            if val1 and val2:
                rows.append({q1.text: val1, q2.text: val2})

        if not rows:
            return None

        df = pd.DataFrame(rows)
        contingency = pd.crosstab(df[q1.text], df[q2.text])

        # Calculate percentages
        pct_by_row = contingency.div(contingency.sum(axis=1), axis=0) * 100
        pct_by_col = contingency.div(contingency.sum(axis=0), axis=1) * 100

        return CrossTabResult(
            question1_id=question1_id,
            question1_text=q1.text,
            question2_id=question2_id,
            question2_text=q2.text,
            contingency_table=contingency,
            percentages_by_row=pct_by_row.fillna(0),
            percentages_by_col=pct_by_col.fillna(0),
        )

    def _get_primary_answer(self, answer: Any, question: Question) -> str | None:
        """Get the primary answer value for cross-tabulation."""
        if answer.text_value:
            return answer.text_value[:50]  # Truncate long text

        if answer.numeric_value is not None:
            return str(int(answer.numeric_value))

        if answer.selected_options:
            # For single-select, return the option
            # For multi-select, join them
            if question.question_type == QuestionType.SINGLE_SELECT:
                return answer.selected_options[0]
            return ", ".join(sorted(answer.selected_options)[:3])  # Limit to 3

        return None

    def _calculate_completion_rate(self) -> float:
        """Calculate overall survey completion rate."""
        if not self.survey.responses:
            return 0.0

        total_questions = len(self.survey.questions)
        if total_questions == 0:
            return 100.0

        completed_counts = []
        for response in self.survey.responses:
            answered = sum(
                1
                for q in self.survey.questions
                if self._has_answer(response.answers.get(q.id))
            )
            completed_counts.append(answered / total_questions * 100)

        return sum(completed_counts) / len(completed_counts)

    def _has_answer(self, answer: Any) -> bool:
        """Check if an answer has content."""
        if not answer:
            return False
        return bool(
            answer.selected_options
            or answer.text_value
            or answer.other_text
            or answer.numeric_value is not None
        )

    def _calculate_avg_completion_time(self) -> float | None:
        """Calculate average survey completion time in minutes."""
        times = []
        for response in self.survey.responses:
            r = response.respondent
            if r.start_date and r.end_date:
                delta = r.end_date - r.start_date
                minutes = delta.total_seconds() / 60
                if 0 < minutes < 120:  # Filter outliers
                    times.append(minutes)

        return sum(times) / len(times) if times else None


def analyze_survey(survey: Survey) -> SurveyAnalysis:
    """Convenience function to analyze a survey.

    Args:
        survey: Parsed Survey object.

    Returns:
        SurveyAnalysis with complete statistics.
    """
    analyzer = SurveyAnalyzer(survey)
    return analyzer.analyze()
