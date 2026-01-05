"""Pydantic data models for survey data structures."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Types of survey questions."""

    SINGLE_SELECT = "single_select"  # Radio buttons - one answer
    MULTI_SELECT = "multi_select"  # Checkboxes - multiple answers
    MATRIX = "matrix"  # Rating grid with sub-questions
    OPEN_TEXT = "open_text"  # Free text response
    NUMERIC_SCALE = "numeric_scale"  # Numeric rating (e.g., -10 to +10)


class QuestionOption(BaseModel):
    """An answer option for a question."""

    label: str
    column_index: int  # Original column index in CSV
    is_other: bool = False  # True if this is an "Other (please specify)" option


class Question(BaseModel):
    """A survey question with its options and responses."""

    id: str  # Unique identifier derived from column position
    text: str  # Question text from header row 1
    question_type: QuestionType
    options: list[QuestionOption] = Field(default_factory=list)
    column_indices: list[int] = Field(default_factory=list)  # All columns for this question
    other_text_column: int | None = None  # Column index for "Other" text responses


class ResponseValue(BaseModel):
    """A single response value for a question."""

    selected_options: list[str] = Field(default_factory=list)  # Selected option labels
    other_text: str | None = None  # Text for "Other" option
    numeric_value: float | None = None  # For numeric scale questions
    text_value: str | None = None  # For open text questions
    raw_values: list[Any] = Field(default_factory=list)  # Original CSV values


class Respondent(BaseModel):
    """A survey respondent with metadata."""

    respondent_id: str
    collector_id: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    ip_address: str | None = None
    email: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    custom_data: str | None = None


class Response(BaseModel):
    """A complete survey response from one respondent."""

    respondent: Respondent
    answers: dict[str, ResponseValue] = Field(default_factory=dict)  # question_id -> response


class Survey(BaseModel):
    """Container for a complete survey with questions and responses."""

    title: str = ""
    questions: list[Question] = Field(default_factory=list)
    responses: list[Response] = Field(default_factory=list)
    metadata_columns: list[str] = Field(default_factory=list)

    @property
    def response_count(self) -> int:
        """Total number of responses."""
        return len(self.responses)

    @property
    def question_count(self) -> int:
        """Total number of questions."""
        return len(self.questions)

    def get_question_by_id(self, question_id: str) -> Question | None:
        """Get a question by its ID."""
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    def get_question_by_text(self, text: str, partial: bool = True) -> Question | None:
        """Get a question by its text (exact or partial match)."""
        text_lower = text.lower()
        for q in self.questions:
            if partial and text_lower in q.text.lower():
                return q
            elif not partial and text_lower == q.text.lower():
                return q
        return None


# ============================================================================
# Insights Configuration Models
# ============================================================================


class QuestionMapping(BaseModel):
    """Maps a question purpose to a question ID or text pattern."""

    question_id: str | None = None  # Direct question ID (e.g., "Q1")
    text_pattern: str | None = None  # Partial text match for auto-detection
    description: str = ""  # Human-readable description of this mapping


class SentimentMapping(BaseModel):
    """Maps sentiment labels to standardized keys for tool importance analysis."""

    super_bummed: str = "Super bummed"
    disappointed: str = "Disappointed"
    neutral: str = "Meh / neutral"
    can_live_without: str = "Can live without"
    good_riddance: str = "Good riddance"


class InsightsConfig(BaseModel):
    """Configuration for insights generation.

    This allows the insights module to work with any survey by mapping
    question purposes to specific question IDs or text patterns.
    """

    # Core question mappings
    department_question: QuestionMapping | None = None
    frequency_question: QuestionMapping | None = None
    tools_question: QuestionMapping | None = None
    use_cases_question: QuestionMapping | None = None
    tool_importance_question: QuestionMapping | None = None
    barriers_question: QuestionMapping | None = None
    cli_tools_question: QuestionMapping | None = None
    used_ai_question: QuestionMapping | None = None

    # Compliance questions
    managed_workspace_question: QuestionMapping | None = None
    identify_admins_question: QuestionMapping | None = None

    # Response value mappings (for custom response text)
    heavy_user_values: list[str] = Field(
        default_factory=lambda: ["More than 3 days per week"]
    )
    yes_values: list[str] = Field(default_factory=lambda: ["Yes"])
    no_values: list[str] = Field(default_factory=lambda: ["No"])
    cli_keywords: list[str] = Field(default_factory=lambda: ["CLI"])
    no_barriers_values: list[str] = Field(default_factory=lambda: ["No barriers"])

    # Sentiment mapping for tool importance
    sentiment_mapping: SentimentMapping = Field(default_factory=SentimentMapping)

    # Thresholds for insight generation
    low_adoption_threshold: float = 50.0  # % below which adoption is considered low
    high_adoption_threshold: float = 60.0  # % above which adoption is considered high
    barrier_alert_threshold: float = 50.0  # % at which barriers trigger warning

    @classmethod
    def create_default(cls) -> "InsightsConfig":
        """Create a default config with common text patterns for auto-detection."""
        return cls(
            department_question=QuestionMapping(
                text_pattern="department",
                description="Question asking about respondent's department",
            ),
            frequency_question=QuestionMapping(
                text_pattern="how often",
                description="Question about usage frequency",
            ),
            tools_question=QuestionMapping(
                text_pattern="which tools",
                description="Question about tools used",
            ),
            use_cases_question=QuestionMapping(
                text_pattern="use case",
                description="Question about use cases",
            ),
            tool_importance_question=QuestionMapping(
                text_pattern="disappointed",
                description="Question about tool importance/stickiness",
            ),
            barriers_question=QuestionMapping(
                text_pattern="barrier",
                description="Question about adoption barriers",
            ),
            cli_tools_question=QuestionMapping(
                text_pattern="cli",
                description="Question about CLI tool usage",
            ),
            used_ai_question=QuestionMapping(
                text_pattern="used ai",
                description="Question about whether respondent has used AI",
            ),
            managed_workspace_question=QuestionMapping(
                text_pattern="workspace",
                description="Question about managed workspace usage",
            ),
            identify_admins_question=QuestionMapping(
                text_pattern="admin",
                description="Question about identifying admins",
            ),
        )
