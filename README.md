# Survey Analyzer

A CLI tool for analyzing and visualizing Survey Monkey CSV exports.

## Features

- Parse Survey Monkey CSV exports with multi-row headers
- Generate summary statistics for all question types
- Cross-tabulate responses between questions
- Create interactive Plotly visualizations
- Generate professional HTML reports

## Installation

```bash
# Clone the repository
git clone https://github.com/tomleelong/SurveyAnalysis.git
cd SurveyAnalysis

# Install with Poetry
poetry install

# Or install with pip
pip install .
```

## Usage

### Generate a Full Report

```bash
survey-analyzer analyze path/to/survey.csv
```

This creates an HTML report in the `output/` directory with:
- Summary statistics
- Interactive charts for each question
- Response rate analysis

### View Summary Statistics

```bash
survey-analyzer summary path/to/survey.csv
```

### List All Questions

```bash
survey-analyzer questions path/to/survey.csv
```

### Cross-Tabulate Questions

```bash
survey-analyzer crosstab path/to/survey.csv --q1 Q1 --q2 Q5
```

### Command Options

```bash
# Specify output file
survey-analyzer analyze survey.csv -o my_report.html

# Include cross-tabulations in report
survey-analyzer analyze survey.csv -x Q1 Q5 -x Q2 Q7

# Generate without charts (faster)
survey-analyzer analyze survey.csv --no-charts

# Verbose output
survey-analyzer analyze survey.csv -v
```

## Supported Question Types

- **Single Select**: Radio button questions
- **Multi Select**: Checkbox questions
- **Matrix/Rating**: Grid questions with rating scales
- **Open Text**: Free text responses
- **Numeric Scale**: Rating scales (-10 to +10, etc.)

## Development

```bash
# Install dev dependencies
poetry install

# Run tests
poetry run pytest

# Format code
poetry run black survey_analyzer/
poetry run isort survey_analyzer/
```

## License

MIT
