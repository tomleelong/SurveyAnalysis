"""Advanced analytics and insights generation for survey data.

This module provides general-purpose survey insights that work with any
Survey Monkey export, without assumptions about specific question content.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field

from .analyzer import SurveyAnalyzer
from .models import InsightsConfig, Question, QuestionMapping, QuestionType, Survey

logger = logging.getLogger(__name__)


@dataclass
class Insight:
    """A single insight or finding from the data."""

    title: str
    description: str
    metric: str
    severity: str = "info"  # info, warning, success
    category: str = "general"


@dataclass
class SegmentProfile:
    """Profile for a segment (e.g., department, role, location)."""

    name: str
    respondent_count: int
    response_rate: float  # % of total respondents
    top_selections: list[tuple[str, int]]  # Top selections from secondary question


@dataclass
class QuestionCorrelation:
    """Correlation between two questions."""

    question1_text: str
    question2_text: str
    segments: list[SegmentProfile]


@dataclass
class ResponseDistribution:
    """Distribution analysis for a question."""

    question_text: str
    question_id: str
    total_responses: int
    top_options: list[tuple[str, int, float]]  # (option, count, percentage)
    concentration: float  # How concentrated responses are (0-100)


@dataclass
class SurveyInsights:
    """Collection of insights from survey analysis."""

    key_insights: list[Insight] = field(default_factory=list)
    segmentation_analysis: list[QuestionCorrelation] = field(default_factory=list)
    response_distributions: list[ResponseDistribution] = field(default_factory=list)
    completion_insights: dict[str, float] = field(default_factory=dict)
    response_patterns: dict[str, int] = field(default_factory=dict)
    # Extended analysis types
    cross_question_analyses: list["CrossQuestionAnalysis"] = field(default_factory=list)
    matrix_sentiment_analyses: list["MatrixSentimentAnalysis"] = field(default_factory=list)
    numeric_distributions: list["NumericDistribution"] = field(default_factory=list)


# ============================================================================
# Extended Analysis Data Structures
# ============================================================================


@dataclass
class SegmentMetric:
    """Metric for a single segment in cross-question analysis."""

    segment_name: str
    total_count: int
    metric_count: int  # Count meeting criteria
    metric_rate: float  # Percentage


@dataclass
class CrossQuestionAnalysis:
    """Analysis correlating two questions (e.g., usage frequency by department)."""

    segment_question_text: str
    metric_question_text: str
    metric_values: list[str]  # Values being measured
    segments: list[SegmentMetric]


@dataclass
class MatrixItemSentiment:
    """Sentiment for a single matrix item (e.g., a tool)."""

    item_name: str
    avg_score: float  # Weighted sentiment score
    response_count: int
    strong_positive_count: int  # Count in top ratings
    strong_positive_pct: float  # % in top ratings
    distribution: dict[str, int] = field(default_factory=dict)  # Full distribution


@dataclass
class MatrixSentimentAnalysis:
    """Sentiment analysis for matrix questions."""

    question_text: str
    sentiment_map: dict[str, int]  # Label -> score mapping
    items: list[MatrixItemSentiment] = field(default_factory=list)


@dataclass
class NumericDistribution:
    """Distribution analysis for numeric scale questions."""

    question_text: str
    question_id: str
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    response_count: int
    distribution: dict[int, int] = field(default_factory=dict)  # Score -> count


class InsightsGenerator:
    """Generate general-purpose insights from survey data.

    This class analyzes survey data to find patterns, distributions,
    and correlations without assuming specific question content.
    """

    def __init__(self, survey: Survey, config: InsightsConfig | None = None):
        """Initialize with survey data and optional config.

        Args:
            survey: The survey data to analyze.
            config: Optional configuration for customizing insights.
        """
        self.survey = survey
        self.analyzer = SurveyAnalyzer(survey)
        self.config = config or InsightsConfig()
        self._question_cache: dict[str, Question | None] = {}

    def _resolve_question(self, mapping: QuestionMapping | None) -> Question | None:
        """Resolve a question mapping to an actual Question object."""
        if mapping is None:
            return None

        cache_key = f"{mapping.question_id}:{mapping.text_pattern}"
        if cache_key in self._question_cache:
            return self._question_cache[cache_key]

        question = None
        if mapping.question_id:
            question = self.survey.get_question_by_id(mapping.question_id)
        if question is None and mapping.text_pattern:
            question = self.survey.get_question_by_text(mapping.text_pattern)

        self._question_cache[cache_key] = question
        return question

    def _resolve_question_by_text(self, text_pattern: str) -> Question | None:
        """Resolve a question by text pattern directly."""
        if text_pattern in self._question_cache:
            return self._question_cache[text_pattern]

        question = self.survey.get_question_by_text(text_pattern)
        self._question_cache[text_pattern] = question
        return question

    def generate(self) -> SurveyInsights:
        """Generate all insights from survey data."""
        insights = SurveyInsights()

        # Analyze completion patterns
        insights.completion_insights = self._analyze_completion()

        # Analyze response distributions for key questions
        insights.response_distributions = self._analyze_distributions()

        # Analyze segmentation if configured
        segment_q = self._resolve_question(self.config.segmentation_question)
        secondary_q = self._resolve_question(self.config.secondary_question)
        if segment_q:
            insights.segmentation_analysis = self._analyze_segmentation(
                segment_q, secondary_q
            )

        # Analyze response patterns
        insights.response_patterns = self._analyze_response_patterns()

        # Generate key insights
        insights.key_insights = self._generate_key_insights(insights)

        return insights

    def _analyze_completion(self) -> dict[str, float]:
        """Analyze survey completion patterns."""
        analysis = self.analyzer.analyze()

        return {
            "completion_rate": analysis.completion_rate,
            "avg_completion_time": analysis.avg_completion_time_minutes or 0,
            "total_responses": analysis.total_responses,
        }

    def _analyze_distributions(self) -> list[ResponseDistribution]:
        """Analyze response distributions for multi-select questions."""
        distributions = []
        analysis = self.analyzer.analyze()

        for stats in analysis.question_stats:
            if not stats.option_counts:
                continue

            # Calculate concentration (how evenly distributed responses are)
            total = sum(stats.option_counts.values())
            if total == 0:
                continue

            # Get top options with percentages
            sorted_options = sorted(
                stats.option_counts.items(), key=lambda x: -x[1]
            )[:5]
            top_options = [
                (opt, count, count / total * 100)
                for opt, count in sorted_options
            ]

            # Calculate concentration using Herfindahl-Hirschman Index
            # Higher = more concentrated, lower = more evenly distributed
            shares = [count / total for count in stats.option_counts.values()]
            hhi = sum(s * s for s in shares) * 100
            concentration = hhi

            distributions.append(
                ResponseDistribution(
                    question_text=stats.question_text,
                    question_id=stats.question_id,
                    total_responses=stats.response_count,
                    top_options=top_options,
                    concentration=concentration,
                )
            )

        # Sort by response count to show most answered questions first
        return sorted(distributions, key=lambda x: -x.total_responses)[:10]

    def _analyze_segmentation(
        self, segment_q: Question, secondary_q: Question | None
    ) -> list[QuestionCorrelation]:
        """Analyze responses segmented by a grouping question."""
        correlations = []
        segment_data: dict[str, dict] = {}

        for response in self.survey.responses:
            answer = response.answers.get(segment_q.id)
            if not answer or not answer.selected_options:
                continue

            # Get secondary question selections if available
            secondary_selections = []
            if secondary_q:
                sec_answer = response.answers.get(secondary_q.id)
                if sec_answer and sec_answer.selected_options:
                    secondary_selections = sec_answer.selected_options

            # Count by segment
            for segment in answer.selected_options:
                if segment not in segment_data:
                    segment_data[segment] = {
                        "count": 0,
                        "selections": Counter(),
                    }
                segment_data[segment]["count"] += 1
                segment_data[segment]["selections"].update(secondary_selections)

        # Build profiles
        total_respondents = self.survey.response_count
        profiles = []
        for name, data in segment_data.items():
            if data["count"] > 0:
                profiles.append(
                    SegmentProfile(
                        name=name,
                        respondent_count=data["count"],
                        response_rate=data["count"] / total_respondents * 100,
                        top_selections=data["selections"].most_common(5),
                    )
                )

        # Sort by respondent count
        profiles = sorted(profiles, key=lambda x: -x.respondent_count)

        if profiles:
            correlations.append(
                QuestionCorrelation(
                    question1_text=segment_q.text,
                    question2_text=secondary_q.text if secondary_q else "",
                    segments=profiles,
                )
            )

        return correlations

    def _analyze_response_patterns(self) -> dict[str, int]:
        """Analyze general response patterns."""
        patterns = {
            "total_questions": self.survey.question_count,
            "single_select_questions": 0,
            "multi_select_questions": 0,
            "open_text_questions": 0,
            "matrix_questions": 0,
        }

        for q in self.survey.questions:
            if q.question_type == QuestionType.SINGLE_SELECT:
                patterns["single_select_questions"] += 1
            elif q.question_type == QuestionType.MULTI_SELECT:
                patterns["multi_select_questions"] += 1
            elif q.question_type == QuestionType.OPEN_TEXT:
                patterns["open_text_questions"] += 1
            elif q.question_type == QuestionType.MATRIX:
                patterns["matrix_questions"] += 1

        return patterns

    def _generate_key_insights(self, insights: SurveyInsights) -> list[Insight]:
        """Generate key insights from analyzed data."""
        key_insights = []

        # 1. Response rate insight
        completion = insights.completion_insights
        if completion.get("completion_rate", 0) > 0:
            rate = completion["completion_rate"]
            severity = "success" if rate >= 80 else "warning" if rate >= 50 else "info"
            key_insights.append(
                Insight(
                    title="Survey Completion",
                    description=f"{rate:.0f}% of respondents completed the survey with {int(completion.get('total_responses', 0))} total responses.",
                    metric=f"{rate:.0f}%",
                    severity=severity,
                    category="completion",
                )
            )

        # 2. Top response concentration
        if insights.response_distributions:
            most_concentrated = max(
                insights.response_distributions,
                key=lambda x: x.concentration,
            )
            if most_concentrated.top_options:
                top_opt, top_count, top_pct = most_concentrated.top_options[0]
                if top_pct > 50:
                    key_insights.append(
                        Insight(
                            title="Dominant Response",
                            description=f'"{top_opt[:40]}..." received {top_pct:.0f}% of responses for "{most_concentrated.question_text[:30]}..."',
                            metric=f"{top_pct:.0f}%",
                            severity="info",
                            category="distribution",
                        )
                    )

        # 3. Segmentation insight
        if insights.segmentation_analysis:
            seg = insights.segmentation_analysis[0]
            if seg.segments:
                largest = seg.segments[0]
                smallest = seg.segments[-1] if len(seg.segments) > 1 else None

                if smallest and largest.respondent_count > smallest.respondent_count * 2:
                    key_insights.append(
                        Insight(
                            title="Segment Imbalance",
                            description=f'"{largest.name}" has {largest.respondent_count} respondents ({largest.response_rate:.0f}%), while "{smallest.name}" has only {smallest.respondent_count} ({smallest.response_rate:.0f}%).',
                            metric=f"{largest.response_rate:.0f}%",
                            severity="warning",
                            category="segmentation",
                        )
                    )

        # 4. Question type breakdown
        patterns = insights.response_patterns
        total_q = patterns.get("total_questions", 0)
        if total_q > 0:
            multi = patterns.get("multi_select_questions", 0)
            single = patterns.get("single_select_questions", 0)
            open_text = patterns.get("open_text_questions", 0)

            key_insights.append(
                Insight(
                    title="Survey Structure",
                    description=f"Survey has {total_q} questions: {single} single-select, {multi} multi-select, {open_text} open-text.",
                    metric=f"{total_q} Q's",
                    severity="info",
                    category="structure",
                )
            )

        # 5. Low response questions
        if insights.response_distributions:
            low_response = [
                d for d in insights.response_distributions
                if d.total_responses < self.survey.response_count * 0.5
            ]
            if low_response:
                key_insights.append(
                    Insight(
                        title="Low Response Questions",
                        description=f"{len(low_response)} question(s) have less than 50% response rate. Consider reviewing question clarity.",
                        metric=f"{len(low_response)}",
                        severity="warning",
                        category="quality",
                    )
                )

        return key_insights

    # =========================================================================
    # Extended Analysis Methods
    # =========================================================================

    def analyze_cross_question(
        self,
        segment_question_text: str,
        metric_question_text: str,
        metric_values: list[str],
    ) -> CrossQuestionAnalysis | None:
        """Analyze a metric question segmented by another question.

        Args:
            segment_question_text: Text pattern for segmentation question (e.g., "department")
            metric_question_text: Text pattern for metric question (e.g., "usage frequency")
            metric_values: Values to count as meeting the metric (e.g., ["More than 3 days"])

        Returns:
            CrossQuestionAnalysis with per-segment metrics, or None if questions not found.
        """
        segment_q = self._resolve_question_by_text(segment_question_text)
        metric_q = self._resolve_question_by_text(metric_question_text)

        if not segment_q or not metric_q:
            logger.warning(
                f"Could not find questions: segment='{segment_question_text}', "
                f"metric='{metric_question_text}'"
            )
            return None

        # Collect data by segment
        segment_data: dict[str, dict] = {}

        for response in self.survey.responses:
            # Get segment value
            seg_answer = response.answers.get(segment_q.id)
            if not seg_answer or not seg_answer.selected_options:
                continue

            # Get metric value
            metric_answer = response.answers.get(metric_q.id)
            has_metric = False
            if metric_answer and metric_answer.selected_options:
                # Check if any selected option matches the metric values
                for opt in metric_answer.selected_options:
                    for mv in metric_values:
                        if mv.lower() in opt.lower():
                            has_metric = True
                            break

            # Count by segment
            for segment in seg_answer.selected_options:
                if segment not in segment_data:
                    segment_data[segment] = {"total": 0, "metric": 0}
                segment_data[segment]["total"] += 1
                if has_metric:
                    segment_data[segment]["metric"] += 1

        # Build segment metrics
        segments = []
        for name, data in segment_data.items():
            if data["total"] > 0:
                rate = (data["metric"] / data["total"]) * 100
                segments.append(
                    SegmentMetric(
                        segment_name=name,
                        total_count=data["total"],
                        metric_count=data["metric"],
                        metric_rate=rate,
                    )
                )

        # Sort by metric rate descending
        segments = sorted(segments, key=lambda x: -x.metric_rate)

        return CrossQuestionAnalysis(
            segment_question_text=segment_q.text,
            metric_question_text=metric_q.text,
            metric_values=metric_values,
            segments=segments,
        )

    def analyze_matrix_sentiment(
        self,
        question_text: str,
        sentiment_map: dict[str, int],
    ) -> MatrixSentimentAnalysis | None:
        """Analyze a matrix question with sentiment scoring.

        Args:
            question_text: Text pattern for matrix question
            sentiment_map: Mapping of option labels to sentiment scores
                          e.g., {"Super bummed": 2, "Disappointed": 1, "Meh": 0}

        Returns:
            MatrixSentimentAnalysis with per-item sentiment scores.
        """
        question = self._resolve_question_by_text(question_text)

        if not question or question.question_type != QuestionType.MATRIX:
            logger.warning(f"Matrix question not found: '{question_text}'")
            return None

        # Parse matrix items from options
        # Matrix options are typically "Item - Rating" format
        item_data: dict[str, dict] = {}

        for opt in question.options:
            # Parse "Item - Rating" format
            if " - " in opt.label:
                parts = opt.label.rsplit(" - ", 1)
                item_name = parts[0].strip()
                rating = parts[1].strip()

                if item_name not in item_data:
                    item_data[item_name] = {"scores": [], "distribution": Counter()}

        # Collect responses
        for response in self.survey.responses:
            answer = response.answers.get(question.id)
            if not answer or not answer.selected_options:
                continue

            for selected in answer.selected_options:
                if " - " in selected:
                    parts = selected.rsplit(" - ", 1)
                    item_name = parts[0].strip()
                    rating = parts[1].strip()

                    if item_name in item_data:
                        # Get sentiment score
                        score = None
                        for sent_label, sent_score in sentiment_map.items():
                            if sent_label.lower() in rating.lower():
                                score = sent_score
                                break

                        if score is not None:
                            item_data[item_name]["scores"].append(score)
                            item_data[item_name]["distribution"][rating] += 1

        # Build item sentiments
        items = []
        max_sentiment = max(sentiment_map.values()) if sentiment_map else 1

        for item_name, data in item_data.items():
            if data["scores"]:
                avg_score = sum(data["scores"]) / len(data["scores"])
                strong_positive = sum(
                    1 for s in data["scores"] if s >= max_sentiment
                )
                items.append(
                    MatrixItemSentiment(
                        item_name=item_name,
                        avg_score=avg_score,
                        response_count=len(data["scores"]),
                        strong_positive_count=strong_positive,
                        strong_positive_pct=(strong_positive / len(data["scores"])) * 100
                        if data["scores"]
                        else 0,
                        distribution=dict(data["distribution"]),
                    )
                )

        # Sort by average score descending
        items = sorted(items, key=lambda x: -x.avg_score)

        return MatrixSentimentAnalysis(
            question_text=question.text,
            sentiment_map=sentiment_map,
            items=items,
        )

    def analyze_numeric_distribution(
        self, question_text: str
    ) -> NumericDistribution | None:
        """Analyze distribution of a numeric scale question.

        Args:
            question_text: Text pattern for the numeric question

        Returns:
            NumericDistribution with statistics and value distribution.
        """
        question = self._resolve_question_by_text(question_text)

        if not question:
            logger.warning(f"Question not found: '{question_text}'")
            return None

        # Collect numeric values
        values = []
        distribution: Counter = Counter()

        for response in self.survey.responses:
            answer = response.answers.get(question.id)
            if not answer:
                continue

            # Try numeric_value first
            if answer.numeric_value is not None:
                val = answer.numeric_value
                values.append(val)
                distribution[int(val)] += 1
            # Fall back to parsing selected options
            elif answer.selected_options:
                for opt in answer.selected_options:
                    # Try to extract number from option
                    import re

                    match = re.search(r"-?\d+", opt)
                    if match:
                        val = float(match.group())
                        values.append(val)
                        distribution[int(val)] += 1
                        break

        if not values:
            return None

        import statistics

        return NumericDistribution(
            question_text=question.text,
            question_id=question.id,
            mean=statistics.mean(values),
            median=statistics.median(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0,
            min_val=min(values),
            max_val=max(values),
            response_count=len(values),
            distribution=dict(distribution),
        )

    def analyze_response_breakdown(
        self, question_text: str
    ) -> ResponseDistribution | None:
        """Get full response breakdown for a specific question.

        Args:
            question_text: Text pattern for the question

        Returns:
            ResponseDistribution with all options and their counts.
        """
        question = self._resolve_question_by_text(question_text)

        if not question:
            logger.warning(f"Question not found: '{question_text}'")
            return None

        # Count responses
        option_counts: Counter = Counter()
        total_responses = 0

        for response in self.survey.responses:
            answer = response.answers.get(question.id)
            if answer and answer.selected_options:
                total_responses += 1
                for opt in answer.selected_options:
                    option_counts[opt] += 1

        if not option_counts:
            return None

        # Calculate percentages
        total_selections = sum(option_counts.values())
        top_options = [
            (opt, count, (count / total_responses) * 100)
            for opt, count in option_counts.most_common()
        ]

        # Calculate concentration
        shares = [count / total_selections for count in option_counts.values()]
        concentration = sum(s * s for s in shares) * 100

        return ResponseDistribution(
            question_text=question.text,
            question_id=question.id,
            total_responses=total_responses,
            top_options=top_options,
            concentration=concentration,
        )
