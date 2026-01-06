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


class InsightsConfig(BaseModel):
    """Configuration for general survey insights.

    This is a simplified configuration that works with any Survey Monkey survey.
    It allows optional segmentation analysis by specifying which questions to use.
    """

    # Segmentation question - use to group respondents (e.g., department, role, location)
    segmentation_question: QuestionMapping | None = None

    # Secondary question - analyze this question's responses by segment
    secondary_question: QuestionMapping | None = None

    # Extended options for generic analysis
    cross_question_pairs: list[tuple[QuestionMapping, QuestionMapping]] = Field(
        default_factory=list
    )
    matrix_sentiment_mappings: dict[str, int] = Field(
        default_factory=dict
    )  # e.g., {"Super bummed": 2, "Disappointed": 1}
    analyze_numeric_distributions: bool = True


# ============================================================================
# Custom Analysis Data Structures
# ============================================================================


class CustomAnalysisConfig(BaseModel):
    """Configuration for a single custom analysis."""

    type: str  # Analysis type: cross_question_metric, matrix_sentiment, etc.
    title: str
    description: str = ""
    question: str | None = None  # Question text pattern for single-question analyses
    segment_question: str | None = None  # For cross-question analyses
    metric_question: str | None = None  # For cross-question analyses
    metric_values: list[str] = Field(default_factory=list)  # Values to count as "metric"
    sentiment_map: dict[str, int] = Field(default_factory=dict)  # For matrix sentiment
    highlight_value: str | None = None  # Value to highlight in charts
    show_stats: bool = True  # Show mean/median for numeric


class CustomConfig(BaseModel):
    """Complete custom analysis configuration."""

    title: str = "Custom Analysis"
    analyses: list[CustomAnalysisConfig] = Field(default_factory=list)
