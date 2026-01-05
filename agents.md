# Agent Instructions

Guidelines for AI agents working on the Survey Analyzer codebase.

## Project Overview

Survey Analyzer is a CLI tool for parsing, analyzing, and visualizing Survey Monkey CSV exports. It generates interactive HTML reports with Plotly charts.

## Architecture

```
survey_analyzer/
├── cli.py          # Click CLI commands (analyze, summary, crosstab, questions)
├── parser.py       # Survey Monkey CSV parser (handles 2-row headers)
├── analyzer.py     # Statistical analysis (summary stats, cross-tabulation)
├── visualizer.py   # Plotly chart generation (SurveyVisualizer, InsightsVisualizer)
├── reporter.py     # HTML report generation with Jinja2
├── insights.py     # Advanced analytics (config-driven, works with any survey)
└── models.py       # Pydantic data models (Survey, Question, Response, InsightsConfig)
```

## Key Design Decisions

### CSV Parser
- Survey Monkey exports have 2-row headers: Row 1 = question text, Row 2 = answer options
- Questions span multiple columns for multi-select and matrix types
- Question types are inferred from header structure (single-select, multi-select, matrix, open-text, numeric-scale)

### Data Flow
1. `parser.py` reads CSV → `Survey` model with `Question` and `Response` objects
2. `analyzer.py` computes statistics → `SurveyAnalysis` with `QuestionStats`
3. `insights.py` generates advanced analytics (optional, config-driven)
4. `visualizer.py` creates Plotly figures from analysis results
5. `reporter.py` renders Jinja2 template with embedded charts

### Insights System
The insights module is config-driven and works with any survey structure:
- Uses `InsightsConfig` model to map question purposes to question IDs or text patterns
- Falls back to auto-detection using common text patterns if no config provided
- Only generates insights for questions that are mapped and found in the survey
- See `examples/bertram_ai_survey_config.json` for a sample config file

## Coding Standards

### Python
- Use type hints for all function parameters and return values
- Follow PEP 8 with Black formatting
- Use Pydantic models for data structures
- Prefer dataclasses for simple data containers without validation
- Use logging module for debug/info messages

### Dependencies
- pandas: Data manipulation
- plotly: Interactive visualizations
- click: CLI framework
- pydantic: Data validation
- jinja2: HTML templating

### Testing
- Run tests with `poetry run pytest`
- Test new parsers with sample CSV files

## Brand Colors (Bertram)

Always use these colors for UI elements and charts:

| Name   | Hex       | Usage                    |
|--------|-----------|--------------------------|
| White  | #FFFFFF   | Card backgrounds         |
| Cream  | #F2EFE9   | Page background          |
| Blue   | #225AA8   | Primary accent, links    |
| Navy   | #232C40   | Secondary, headers       |
| Royal  | #1C2334   | Text, dark accents       |

Colors are defined in:
- `visualizer.py`: `BERTRAM_COLORS` dict and `COLORS` list
- `templates/report.html`: CSS custom properties (`:root`)

## Common Tasks

### Adding a New Chart Type
1. Add method to `SurveyVisualizer` class in `visualizer.py`
2. Use `COLORS` list for consistent branding
3. Set `template=CHART_TEMPLATE` in layout

### Adding a New CLI Command
1. Add function with `@cli.command()` decorator in `cli.py`
2. Use `click.argument()` and `click.option()` for parameters
3. Follow existing patterns for error handling and output

### Modifying the Report Template
1. Edit `templates/report.html`
2. Use Jinja2 syntax for dynamic content
3. Keep CSS variables in `:root` for theme consistency

### Adding a New Question Type
1. Add enum value to `QuestionType` in `models.py`
2. Update `_infer_question_type()` in `parser.py`
3. Handle in `_parse_question_response()` in `parser.py`
4. Add visualization support in `visualizer.py` if needed

## File Locations

- Sample data: Project root (`*.csv` files)
- Generated reports: `output/` directory
- HTML template: `templates/report.html`
- Tests: `tests/` directory

## Commands

```bash
# Install dependencies
poetry install

# Run CLI
poetry run survey-analyzer --help
poetry run survey-analyzer analyze data.csv
poetry run survey-analyzer summary data.csv
poetry run survey-analyzer questions data.csv
poetry run survey-analyzer crosstab data.csv --q1 Q1 --q2 Q5

# With custom insights config
poetry run survey-analyzer analyze data.csv --insights-config config.json

# Without insights (basic report only)
poetry run survey-analyzer analyze data.csv --no-insights

# Run tests
poetry run pytest

# Format code
poetry run black survey_analyzer/
poetry run isort survey_analyzer/
```

## Insights Configuration

The insights system uses a JSON config file to map question purposes to question IDs. This makes the advanced analytics work with any survey structure.

### Config File Structure
```json
{
  "department_question": {"question_id": "Q1"},
  "frequency_question": {"question_id": "Q5"},
  "tools_question": {"question_id": "Q6"},
  "barriers_question": {"text_pattern": "barrier"},
  "heavy_user_values": ["More than 3 days per week"],
  "low_adoption_threshold": 50.0
}
```

### Question Mappings
Each mapping can use either:
- `question_id`: Direct ID like "Q1", "Q2", etc.
- `text_pattern`: Partial text match for auto-detection

Available mappings:
- `department_question`: Respondent's department/team
- `frequency_question`: Usage frequency
- `tools_question`: Tools used
- `use_cases_question`: Use cases
- `tool_importance_question`: Tool stickiness/disappointment
- `barriers_question`: Adoption barriers
- `cli_tools_question`: CLI tool usage
- `used_ai_question`: Whether respondent has used AI
- `managed_workspace_question`: Workspace compliance
- `identify_admins_question`: Admin identification

### Response Value Mappings
- `heavy_user_values`: Values indicating heavy usage
- `yes_values`: Affirmative responses
- `no_values`: Negative responses
- `cli_keywords`: Keywords identifying CLI tools
- `no_barriers_values`: Values indicating no barriers
