"""Survey Monkey CSV parser with multi-row header handling."""

import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from .models import (
    Question,
    QuestionOption,
    QuestionType,
    Respondent,
    Response,
    ResponseValue,
    Survey,
)

logger = logging.getLogger(__name__)

# Standard Survey Monkey metadata columns (first 9 columns typically)
METADATA_COLUMNS = [
    "Respondent ID",
    "Collector ID",
    "Start Date",
    "End Date",
    "IP Address",
    "Email Address",
    "First Name",
    "Last Name",
    "Custom Data 1",
]


class SurveyMonkeyParser:
    """Parser for Survey Monkey CSV exports."""

    def __init__(self, file_path: str | Path):
        """Initialize parser with file path.

        Args:
            file_path: Path to the Survey Monkey CSV export file.
        """
        self.file_path = Path(file_path)
        self._raw_df: pd.DataFrame | None = None
        self._header_row1: list[str] = []
        self._header_row2: list[str] = []

    def parse(self) -> Survey:
        """Parse the CSV file and return a Survey object.

        Returns:
            Survey object containing questions and responses.
        """
        self._load_csv()
        questions = self._parse_questions()
        responses = self._parse_responses(questions)

        # Extract title from filename
        title = self.file_path.stem

        return Survey(
            title=title,
            questions=questions,
            responses=responses,
            metadata_columns=METADATA_COLUMNS,
        )

    def _load_csv(self) -> None:
        """Load the CSV file and extract headers."""
        # Read raw CSV to get both header rows
        with open(self.file_path, encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) < 3:
            raise ValueError("CSV file must have at least 2 header rows and 1 data row")

        # Parse header rows (handling quoted commas)
        self._header_row1 = self._parse_csv_line(lines[0])
        self._header_row2 = self._parse_csv_line(lines[1])

        # Load data starting from row 3 (index 2)
        self._raw_df = pd.read_csv(
            self.file_path,
            skiprows=2,
            header=None,
            dtype=str,
            keep_default_na=False,
        )

        # Ensure column count matches headers
        while len(self._header_row1) < len(self._raw_df.columns):
            self._header_row1.append("")
        while len(self._header_row2) < len(self._raw_df.columns):
            self._header_row2.append("")

        logger.info(f"Loaded {len(self._raw_df)} responses with {len(self._header_row1)} columns")

    def _parse_csv_line(self, line: str) -> list[str]:
        """Parse a CSV line handling quoted fields."""
        result = []
        current = ""
        in_quotes = False

        for char in line.strip():
            if char == '"':
                in_quotes = not in_quotes
            elif char == "," and not in_quotes:
                result.append(current.strip())
                current = ""
            else:
                current += char

        result.append(current.strip())
        return result

    def _parse_questions(self) -> list[Question]:
        """Parse question structure from headers."""
        questions: list[Question] = []
        current_question: Question | None = None
        question_idx = 0

        # Skip metadata columns
        start_col = len(METADATA_COLUMNS)

        col = start_col
        while col < len(self._header_row1):
            q_text = self._header_row1[col].strip()
            option_text = self._header_row2[col].strip()

            # New question starts when row1 has text
            if q_text:
                # Save previous question if exists
                if current_question:
                    current_question.question_type = self._infer_question_type(current_question)
                    questions.append(current_question)

                # Start new question
                question_idx += 1
                current_question = Question(
                    id=f"Q{question_idx}",
                    text=q_text,
                    question_type=QuestionType.SINGLE_SELECT,  # Default, will be updated
                    column_indices=[col],
                )

                # Add first option if exists
                if option_text:
                    is_other = self._is_other_option(option_text)
                    current_question.options.append(
                        QuestionOption(
                            label=option_text,
                            column_index=col,
                            is_other=is_other,
                        )
                    )
                    if is_other:
                        # Next column might be the "Other" text field
                        current_question.other_text_column = col

            elif current_question and option_text:
                # Continuation of current question (multi-select or matrix)
                current_question.column_indices.append(col)
                is_other = self._is_other_option(option_text)
                current_question.options.append(
                    QuestionOption(
                        label=option_text,
                        column_index=col,
                        is_other=is_other,
                    )
                )
                if is_other:
                    current_question.other_text_column = col

            col += 1

        # Don't forget the last question
        if current_question:
            current_question.question_type = self._infer_question_type(current_question)
            questions.append(current_question)

        logger.info(f"Parsed {len(questions)} questions")
        return questions

    def _is_other_option(self, text: str) -> bool:
        """Check if an option is an 'Other' option."""
        text_lower = text.lower()
        return "other" in text_lower and ("specify" in text_lower or "please" in text_lower)

    def _infer_question_type(self, question: Question) -> QuestionType:
        """Infer question type from structure and options."""
        # Check if it's an open-ended question (single column, no options or generic option)
        if len(question.options) <= 1:
            option_text = question.options[0].label if question.options else ""
            if not option_text or option_text.lower() in ["open-ended response", "response"]:
                return QuestionType.OPEN_TEXT

        # Check for numeric scale (options look like numbers or rating descriptions)
        if question.options:
            numeric_pattern = re.compile(r"^-?\d+$|^\s*-\s*\d+$")
            if all(
                numeric_pattern.match(opt.label.strip()) or opt.label.strip().startswith(" - ")
                for opt in question.options
                if opt.label.strip()
            ):
                return QuestionType.NUMERIC_SCALE

        # Check for matrix (options have consistent patterns like "Item - Rating")
        if len(question.options) > 4:
            # Look for pattern like "Product A - Satisfied", "Product A - Neutral"
            dash_pattern = re.compile(r"^.+ - .+$")
            if all(dash_pattern.match(opt.label) for opt in question.options if opt.label):
                return QuestionType.MATRIX

        # Multiple columns = multi-select, single column = single-select
        if len(question.column_indices) > 1:
            return QuestionType.MULTI_SELECT

        return QuestionType.SINGLE_SELECT

    def _parse_responses(self, questions: list[Question]) -> list[Response]:
        """Parse response data."""
        if self._raw_df is None:
            return []

        responses: list[Response] = []

        for _, row in self._raw_df.iterrows():
            respondent = self._parse_respondent(row)
            answers = self._parse_answers(row, questions)

            responses.append(Response(respondent=respondent, answers=answers))

        return responses

    def _parse_respondent(self, row: pd.Series) -> Respondent:
        """Parse respondent metadata from a row."""
        def safe_get(idx: int) -> str:
            try:
                val = row.iloc[idx] if idx < len(row) else ""
                return str(val).strip() if pd.notna(val) else ""
            except (IndexError, KeyError):
                return ""

        def parse_date(date_str: str) -> datetime | None:
            if not date_str:
                return None
            try:
                return datetime.strptime(date_str, "%m/%d/%Y %I:%M:%S %p")
            except ValueError:
                try:
                    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    return None

        return Respondent(
            respondent_id=safe_get(0),
            collector_id=safe_get(1) or None,
            start_date=parse_date(safe_get(2)),
            end_date=parse_date(safe_get(3)),
            ip_address=safe_get(4) or None,
            email=safe_get(5) or None,
            first_name=safe_get(6) or None,
            last_name=safe_get(7) or None,
            custom_data=safe_get(8) or None,
        )

    def _parse_answers(
        self, row: pd.Series, questions: list[Question]
    ) -> dict[str, ResponseValue]:
        """Parse answers for all questions from a row."""
        answers: dict[str, ResponseValue] = {}

        for question in questions:
            response_value = self._parse_question_response(row, question)
            answers[question.id] = response_value

        return answers

    def _parse_question_response(
        self, row: pd.Series, question: Question
    ) -> ResponseValue:
        """Parse response for a single question."""
        raw_values = []
        selected_options = []
        other_text = None
        numeric_value = None
        text_value = None

        for opt in question.options:
            try:
                cell_value = row.iloc[opt.column_index]
                raw_values.append(cell_value)

                if pd.notna(cell_value) and str(cell_value).strip():
                    cell_str = str(cell_value).strip()

                    if opt.is_other:
                        # For "Other" options, the value might be the text itself
                        if not cell_str.isdigit():
                            other_text = cell_str
                        selected_options.append(opt.label)
                    elif cell_str.isdigit() or cell_str.lstrip("-").isdigit():
                        # Numeric indicator that option was selected
                        selected_options.append(opt.label)
                    else:
                        # Text value (could be other text or open response)
                        if question.question_type == QuestionType.OPEN_TEXT:
                            text_value = cell_str
                        else:
                            other_text = cell_str

            except (IndexError, KeyError):
                continue

        # Handle numeric scale questions
        if question.question_type == QuestionType.NUMERIC_SCALE and selected_options:
            # Try to extract numeric value from first selected option
            try:
                # Parse from option label (e.g., " - 5" -> 5)
                num_match = re.search(r"-?\d+", selected_options[0])
                if num_match:
                    numeric_value = float(num_match.group())
            except (ValueError, IndexError):
                pass

        return ResponseValue(
            selected_options=selected_options,
            other_text=other_text,
            numeric_value=numeric_value,
            text_value=text_value,
            raw_values=raw_values,
        )


def parse_survey(file_path: str | Path) -> Survey:
    """Convenience function to parse a Survey Monkey CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Survey object with parsed questions and responses.
    """
    parser = SurveyMonkeyParser(file_path)
    return parser.parse()
