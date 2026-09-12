# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- Start server: `python manage.py runserver`
- Start server on specific port: `python manage.py runserver 8080`

### Testing
- Run all tests: `python manage.py test`
- Run tests for a specific app: `python manage.py test loader`
- Run a specific test class: `python manage.py test loader.tests.TestClearSpendingClient`

## Architecture Overview

The project is a Django-based application designed to analyze government procurement contracts from the ClearSpending API.

### High-Level Structure
- **API Client (`loader/api_client.py`)**: A low-level wrapper for the ClearSpending REST API. Handles authentication (API keys), request formatting (including mandatory `format=json`), and HTTP error handling.
- **Service Layer (`loader/services.py`)**: Contains the core business logic. It orchestrates the data flow:
    1. Fetches a list of contracts based on filters (Stage 1).
    2. Fetches detailed data for each contract using its `regnum` (Stage 2).
    3. Processes raw API data into structured Pandas DataFrames.
- **Analytics Layer (`loader/analytics.py`)**: Performs data analysis, OKPD2 classification, and statistical processing using `pandas` and `scikit-learn`.
- **View Layer (`loader/views.py`)**: Django views that handle user input via `UserForm` and render results using templates.
- **Templates (`templates/`)**: HTML templates for the user interface, utilizing Bootstrap for layout.

### Data Flow
`User Input` $\rightarrow$ `views.py` $\rightarrow$ `services.py` $\rightarrow$ `api_client.py` $\rightarrow$ `ClearSpending API` $\rightarrow$ `analytics.py` $\rightarrow$ `templates/`
