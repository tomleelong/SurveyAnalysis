"""Advanced analytics and insights generation for survey data."""

import logging
from collections import Counter
from dataclasses import dataclass, field

from .analyzer import SurveyAnalyzer
from .models import InsightsConfig, Question, QuestionMapping, Survey

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
class DepartmentProfile:
    """Usage profile for a department."""

    name: str
    respondent_count: int
    heavy_users_pct: float
    top_tools: list[tuple[str, int]]
    top_use_cases: list[tuple[str, int]]


@dataclass
class ToolImportance:
    """Importance metrics for a tool."""

    name: str
    user_count: int
    super_bummed: int
    disappointed: int
    neutral: int
    can_live_without: int
    good_riddance: int

    @property
    def stickiness_score(self) -> float:
        """Calculate stickiness as % who would miss it."""
        total = self.super_bummed + self.disappointed + self.neutral + self.can_live_without + self.good_riddance
        if total == 0:
            return 0
        return (self.super_bummed + self.disappointed) / total * 100


@dataclass
class SurveyInsights:
    """Collection of insights from survey analysis."""

    key_insights: list[Insight] = field(default_factory=list)
    department_profiles: list[DepartmentProfile] = field(default_factory=list)
    tool_importance: list[ToolImportance] = field(default_factory=list)
    barriers: dict[str, int] = field(default_factory=dict)
    compliance_gap: dict[str, int] = field(default_factory=dict)
    adoption_funnel: dict[str, int] = field(default_factory=dict)


class InsightsGenerator:
    """Generate advanced insights from survey data.

    This class is config-driven and can work with any survey structure
    by mapping question purposes to question IDs or text patterns.
    """

    def __init__(self, survey: Survey, config: InsightsConfig | None = None):
        """Initialize with survey data and optional config.

        Args:
            survey: The survey data to analyze.
            config: Configuration mapping question purposes to IDs.
                   If None, attempts auto-detection using default patterns.
        """
        self.survey = survey
        self.analyzer = SurveyAnalyzer(survey)
        self.config = config or InsightsConfig.create_default()
        self._question_cache: dict[str, Question | None] = {}

    def _resolve_question(self, mapping: QuestionMapping | None) -> Question | None:
        """Resolve a question mapping to an actual Question object.

        Args:
            mapping: The question mapping to resolve.

        Returns:
            The matched Question or None if not found.
        """
        if mapping is None:
            return None

        # Create cache key
        cache_key = f"{mapping.question_id}:{mapping.text_pattern}"
        if cache_key in self._question_cache:
            return self._question_cache[cache_key]

        question = None

        # Try direct ID first
        if mapping.question_id:
            question = self.survey.get_question_by_id(mapping.question_id)

        # Fall back to text pattern matching
        if question is None and mapping.text_pattern:
            question = self.survey.get_question_by_text(mapping.text_pattern)

        self._question_cache[cache_key] = question
        return question

    def _get_response_options(
        self, response, question: Question | None
    ) -> list[str]:
        """Get selected options for a question from a response."""
        if question is None:
            return []
        answer = response.answers.get(question.id)
        return answer.selected_options if answer else []

    def _matches_any(self, options: list[str], values: list[str]) -> bool:
        """Check if any option matches any of the target values."""
        return any(v in options for v in values)

    def _contains_any_keyword(self, options: list[str], keywords: list[str]) -> bool:
        """Check if any option contains any of the keywords."""
        for opt in options:
            for kw in keywords:
                if kw.lower() in opt.lower():
                    return True
        return False

    def generate(self) -> SurveyInsights:
        """Generate all insights based on available question mappings."""
        insights = SurveyInsights()

        # Only generate insights for questions that are mapped and found
        dept_q = self._resolve_question(self.config.department_question)
        freq_q = self._resolve_question(self.config.frequency_question)
        tools_q = self._resolve_question(self.config.tools_question)
        usecases_q = self._resolve_question(self.config.use_cases_question)

        if dept_q:
            insights.department_profiles = self._analyze_departments(
                dept_q, freq_q, tools_q, usecases_q
            )

        tool_importance_q = self._resolve_question(self.config.tool_importance_question)
        if tool_importance_q and tools_q:
            insights.tool_importance = self._analyze_tool_importance(
                tool_importance_q, tools_q
            )

        barriers_q = self._resolve_question(self.config.barriers_question)
        if barriers_q:
            insights.barriers = self._analyze_barriers(barriers_q)

        workspace_q = self._resolve_question(self.config.managed_workspace_question)
        admins_q = self._resolve_question(self.config.identify_admins_question)
        if workspace_q or admins_q:
            insights.compliance_gap = self._analyze_compliance(workspace_q, admins_q)

        used_ai_q = self._resolve_question(self.config.used_ai_question)
        cli_q = self._resolve_question(self.config.cli_tools_question)
        insights.adoption_funnel = self._build_adoption_funnel(
            used_ai_q, freq_q, tools_q, cli_q
        )

        insights.key_insights = self._generate_key_insights(insights)

        return insights

    def _analyze_departments(
        self,
        dept_q: Question,
        freq_q: Question | None,
        tools_q: Question | None,
        usecases_q: Question | None,
    ) -> list[DepartmentProfile]:
        """Analyze usage patterns by department."""
        dept_data: dict[str, dict] = {}

        for response in self.survey.responses:
            dept_names = self._get_response_options(response, dept_q)
            if not dept_names:
                continue

            freq_options = self._get_response_options(response, freq_q)
            is_heavy = self._matches_any(freq_options, self.config.heavy_user_values)

            tool_names = self._get_response_options(response, tools_q)
            usecase_names = self._get_response_options(response, usecases_q)

            for dept in dept_names:
                if dept not in dept_data:
                    dept_data[dept] = {
                        "count": 0,
                        "heavy": 0,
                        "tools": Counter(),
                        "usecases": Counter(),
                    }
                dept_data[dept]["count"] += 1
                if is_heavy:
                    dept_data[dept]["heavy"] += 1
                dept_data[dept]["tools"].update(tool_names)
                dept_data[dept]["usecases"].update(usecase_names)

        profiles = []
        for name, data in dept_data.items():
            if data["count"] > 0:
                profiles.append(
                    DepartmentProfile(
                        name=name,
                        respondent_count=data["count"],
                        heavy_users_pct=data["heavy"] / data["count"] * 100,
                        top_tools=data["tools"].most_common(5),
                        top_use_cases=data["usecases"].most_common(5),
                    )
                )

        return sorted(profiles, key=lambda x: -x.heavy_users_pct)

    def _analyze_tool_importance(
        self, importance_q: Question, tools_q: Question
    ) -> list[ToolImportance]:
        """Analyze tool stickiness from disappointment/importance question."""
        q_stats = self.analyzer.analyze_question(importance_q)
        tools_stats = self.analyzer.analyze_question(tools_q)

        tool_data: dict[str, dict] = {}
        sentiment = self.config.sentiment_mapping
        sentiment_map = {
            sentiment.super_bummed: "super_bummed",
            sentiment.disappointed: "disappointed",
            sentiment.neutral: "neutral",
            sentiment.can_live_without: "can_live_without",
            sentiment.good_riddance: "good_riddance",
        }

        for opt, count in q_stats.option_counts.items():
            if " - " in opt:
                parts = opt.rsplit(" - ", 1)
                if len(parts) == 2:
                    tool, sentiment_label = parts
                    if tool not in tool_data:
                        tool_data[tool] = {
                            "super_bummed": 0,
                            "disappointed": 0,
                            "neutral": 0,
                            "can_live_without": 0,
                            "good_riddance": 0,
                        }
                    key = sentiment_map.get(sentiment_label)
                    if key:
                        tool_data[tool][key] = count

        tools = []
        for name, data in tool_data.items():
            user_count = tools_stats.option_counts.get(name, 0)
            tools.append(
                ToolImportance(
                    name=name,
                    user_count=user_count,
                    **data,
                )
            )

        return sorted(tools, key=lambda x: -x.stickiness_score)

    def _analyze_barriers(self, barriers_q: Question) -> dict[str, int]:
        """Analyze barriers to adoption."""
        q_stats = self.analyzer.analyze_question(barriers_q)
        return dict(sorted(q_stats.option_counts.items(), key=lambda x: -x[1]))

    def _analyze_compliance(
        self, workspace_q: Question | None, admins_q: Question | None
    ) -> dict[str, int]:
        """Analyze workspace compliance gaps."""
        result = {}

        if workspace_q:
            ws_stats = self.analyzer.analyze_question(workspace_q)
            yes_count = sum(
                ws_stats.option_counts.get(v, 0) for v in self.config.yes_values
            )
            no_count = sum(
                ws_stats.option_counts.get(v, 0) for v in self.config.no_values
            )
            result["using_managed_workspace"] = yes_count
            result["not_using_managed_workspace"] = no_count

        if admins_q:
            admin_stats = self.analyzer.analyze_question(admins_q)
            yes_count = sum(
                admin_stats.option_counts.get(v, 0) for v in self.config.yes_values
            )
            no_count = sum(
                admin_stats.option_counts.get(v, 0) for v in self.config.no_values
            )
            result["can_identify_admins"] = yes_count
            result["cannot_identify_admins"] = no_count

        return result

    def _build_adoption_funnel(
        self,
        used_ai_q: Question | None,
        freq_q: Question | None,
        tools_q: Question | None,
        cli_q: Question | None,
    ) -> dict[str, int]:
        """Build adoption funnel metrics."""
        total = self.survey.response_count

        used_ai = 0
        heavy_users = 0
        multi_tool = 0
        cli_users = 0

        for response in self.survey.responses:
            # Has used AI
            if used_ai_q:
                opts = self._get_response_options(response, used_ai_q)
                if self._matches_any(opts, self.config.yes_values):
                    used_ai += 1

            # Heavy user
            if freq_q:
                opts = self._get_response_options(response, freq_q)
                if self._matches_any(opts, self.config.heavy_user_values):
                    heavy_users += 1

            # Multi-tool user (3+ tools)
            if tools_q:
                opts = self._get_response_options(response, tools_q)
                if len(opts) >= 3:
                    multi_tool += 1

            # CLI user
            if cli_q:
                opts = self._get_response_options(response, cli_q)
                if self._contains_any_keyword(opts, self.config.cli_keywords):
                    cli_users += 1

        return {
            "total_respondents": total,
            "used_ai_at_work": used_ai,
            "heavy_users": heavy_users,
            "multi_tool_users": multi_tool,
            "cli_power_users": cli_users,
        }

    def _generate_key_insights(self, insights: SurveyInsights) -> list[Insight]:
        """Generate key insights from analyzed data."""
        key_insights = []

        # 1. Department adoption gap
        if insights.department_profiles:
            top_dept = insights.department_profiles[0]
            bottom_depts = [
                d
                for d in insights.department_profiles
                if d.heavy_users_pct < self.config.low_adoption_threshold
            ]
            if bottom_depts:
                key_insights.append(
                    Insight(
                        title="Department Adoption Gap",
                        description=f"{top_dept.name} leads with {top_dept.heavy_users_pct:.0f}% heavy users, while {len(bottom_depts)} department(s) have <{self.config.low_adoption_threshold:.0f}% adoption.",
                        metric=f"{top_dept.heavy_users_pct:.0f}% vs <{self.config.low_adoption_threshold:.0f}%",
                        severity="warning",
                        category="adoption",
                    )
                )

        # 2. Tool stickiness
        if insights.tool_importance:
            stickiest = insights.tool_importance[0]
            key_insights.append(
                Insight(
                    title="Most Critical Tool",
                    description=f"{stickiest.name} has {stickiest.stickiness_score:.0f}% stickiness - {stickiest.super_bummed} would be 'super bummed' to lose it.",
                    metric=f"{stickiest.stickiness_score:.0f}%",
                    severity="success",
                    category="tools",
                )
            )

        # 3. Barriers insight
        if insights.barriers:
            no_barriers = sum(
                insights.barriers.get(v, 0) for v in self.config.no_barriers_values
            )
            total = sum(insights.barriers.values())
            has_barriers_pct = (1 - no_barriers / total) * 100 if total > 0 else 0
            if has_barriers_pct > self.config.barrier_alert_threshold:
                other_barriers = [
                    k for k in insights.barriers.keys() if k not in self.config.no_barriers_values
                ]
                top_barrier = other_barriers[0] if other_barriers else "Unknown"
                key_insights.append(
                    Insight(
                        title="Adoption Barriers Exist",
                        description=f"{has_barriers_pct:.0f}% report at least one barrier. Top barrier: '{top_barrier}'",
                        metric=f"{has_barriers_pct:.0f}%",
                        severity="warning",
                        category="barriers",
                    )
                )

        # 4. Compliance gap
        cannot_identify = insights.compliance_gap.get("cannot_identify_admins", 0)
        can_identify = insights.compliance_gap.get("can_identify_admins", 0)
        if cannot_identify > 0:
            gap_pct = cannot_identify / (cannot_identify + can_identify) * 100
            key_insights.append(
                Insight(
                    title="Governance Gap",
                    description=f"{cannot_identify} users ({gap_pct:.0f}%) cannot identify their workspace admins - potential compliance risk.",
                    metric=f"{cannot_identify} users",
                    severity="warning",
                    category="compliance",
                )
            )

        # 5. Adoption funnel
        funnel = insights.adoption_funnel
        if funnel:
            total = funnel.get("total_respondents", 1)
            heavy = funnel.get("heavy_users", 0)
            heavy_pct = heavy / total * 100 if total > 0 else 0
            key_insights.append(
                Insight(
                    title="Heavy User Rate",
                    description=f"{heavy_pct:.0f}% of respondents use AI >3 days/week.",
                    metric=f"{heavy_pct:.0f}%",
                    severity="success" if heavy_pct > self.config.high_adoption_threshold else "info",
                    category="adoption",
                )
            )

        return key_insights
