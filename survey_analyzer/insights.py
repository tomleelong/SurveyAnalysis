"""Advanced analytics and insights generation for survey data."""

from collections import Counter
from dataclasses import dataclass, field

from .analyzer import SurveyAnalyzer
from .models import Survey


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
    """Generate advanced insights from survey data."""

    def __init__(self, survey: Survey):
        """Initialize with survey data."""
        self.survey = survey
        self.analyzer = SurveyAnalyzer(survey)

    def generate(self) -> SurveyInsights:
        """Generate all insights."""
        insights = SurveyInsights()

        insights.department_profiles = self._analyze_departments()
        insights.tool_importance = self._analyze_tool_importance()
        insights.barriers = self._analyze_barriers()
        insights.compliance_gap = self._analyze_compliance()
        insights.adoption_funnel = self._build_adoption_funnel()
        insights.key_insights = self._generate_key_insights(insights)

        return insights

    def _analyze_departments(self) -> list[DepartmentProfile]:
        """Analyze usage patterns by department."""
        dept_data: dict[str, dict] = {}

        for response in self.survey.responses:
            depts = response.answers.get("Q1", None)
            if not depts:
                continue
            dept_names = depts.selected_options

            freq = response.answers.get("Q5", None)
            freq_options = freq.selected_options if freq else []
            is_heavy = "More than 3 days per week" in freq_options

            tools = response.answers.get("Q6", None)
            tool_names = tools.selected_options if tools else []

            usecases = response.answers.get("Q9", None)
            usecase_names = usecases.selected_options if usecases else []

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

    def _analyze_tool_importance(self) -> list[ToolImportance]:
        """Analyze tool stickiness from disappointment question."""
        q11 = self.analyzer.analyze_question(self.survey.get_question_by_id("Q11"))

        tool_data: dict[str, dict] = {}
        sentiment_map = {
            "Super bummed": "super_bummed",
            "Disappointed": "disappointed",
            "Meh / neutral": "neutral",
            "Can live without": "can_live_without",
            "Good riddance": "good_riddance",
        }

        for opt, count in q11.option_counts.items():
            if " - " in opt:
                parts = opt.rsplit(" - ", 1)
                if len(parts) == 2:
                    tool, sentiment = parts
                    if tool not in tool_data:
                        tool_data[tool] = {
                            "super_bummed": 0,
                            "disappointed": 0,
                            "neutral": 0,
                            "can_live_without": 0,
                            "good_riddance": 0,
                        }
                    key = sentiment_map.get(sentiment)
                    if key:
                        tool_data[tool][key] = count

        # Get user counts from Q6
        q6 = self.analyzer.analyze_question(self.survey.get_question_by_id("Q6"))

        tools = []
        for name, data in tool_data.items():
            user_count = q6.option_counts.get(name, 0)
            tools.append(
                ToolImportance(
                    name=name,
                    user_count=user_count,
                    **data,
                )
            )

        return sorted(tools, key=lambda x: -x.stickiness_score)

    def _analyze_barriers(self) -> dict[str, int]:
        """Analyze barriers to AI adoption."""
        q14 = self.analyzer.analyze_question(self.survey.get_question_by_id("Q14"))
        return dict(sorted(q14.option_counts.items(), key=lambda x: -x[1]))

    def _analyze_compliance(self) -> dict[str, int]:
        """Analyze workspace compliance gaps."""
        q7 = self.analyzer.analyze_question(self.survey.get_question_by_id("Q7"))
        q8 = self.analyzer.analyze_question(self.survey.get_question_by_id("Q8"))

        return {
            "using_managed_workspace": q7.option_counts.get("Yes", 0),
            "not_using_managed_workspace": q7.option_counts.get("No", 0),
            "can_identify_admins": q8.option_counts.get("Yes", 0),
            "cannot_identify_admins": q8.option_counts.get("No", 0),
        }

    def _build_adoption_funnel(self) -> dict[str, int]:
        """Build adoption funnel metrics."""
        total = self.survey.response_count

        # Count various adoption stages
        used_ai = 0
        heavy_users = 0
        multi_tool = 0
        cli_users = 0

        for response in self.survey.responses:
            # Has used AI at work
            q4 = response.answers.get("Q4")
            if q4 and "Yes" in q4.selected_options:
                used_ai += 1

            # Heavy user
            q5 = response.answers.get("Q5")
            if q5 and "More than 3 days per week" in q5.selected_options:
                heavy_users += 1

            # Multi-tool user (3+ tools)
            q6 = response.answers.get("Q6")
            if q6 and len(q6.selected_options) >= 3:
                multi_tool += 1

            # CLI user
            q10 = response.answers.get("Q10")
            if q10:
                cli_tools = [t for t in q10.selected_options if "CLI" in t]
                if cli_tools:
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
            bottom_depts = [d for d in insights.department_profiles if d.heavy_users_pct < 50]
            if bottom_depts:
                key_insights.append(
                    Insight(
                        title="Department Adoption Gap",
                        description=f"{top_dept.name} leads with {top_dept.heavy_users_pct:.0f}% heavy users, while {len(bottom_depts)} department(s) have <50% adoption.",
                        metric=f"{top_dept.heavy_users_pct:.0f}% vs <50%",
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
            no_barriers = insights.barriers.get("No barriers", 0)
            total = sum(insights.barriers.values())
            has_barriers_pct = (1 - no_barriers / total) * 100 if total > 0 else 0
            if has_barriers_pct > 50:
                top_barrier = [k for k in insights.barriers.keys() if k != "No barriers"][0]
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
            heavy_pct = heavy / total * 100
            key_insights.append(
                Insight(
                    title="Heavy User Rate",
                    description=f"{heavy_pct:.0f}% of respondents use AI >3 days/week.",
                    metric=f"{heavy_pct:.0f}%",
                    severity="success" if heavy_pct > 60 else "info",
                    category="adoption",
                )
            )

        return key_insights
