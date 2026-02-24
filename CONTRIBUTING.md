# Contributing

Thanks for your interest in contributing to DocAI Assistant!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/dome317/doc-ai-assistant.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `streamlit run app.py`
5. Run tests: `pytest tests/ -v`

## Development

### Code Style

- Python 3.11+
- Use [ruff](https://github.com/astral-sh/ruff) for linting
- Type hints where practical
- Keep functions under 50 lines
- Keep files under 800 lines

### Testing

- Write tests for new features
- Minimum 80% test coverage
- Run: `pytest tests/ -v`
- For live API tests: `ANTHROPIC_API_KEY=sk-ant-... pytest tests/ -v -m live`

### Adding Your Own Product Catalog

1. Edit `products.json` following the existing schema
2. Each product needs: `id`, `name`, `category`, `description`, `specs`, `use_cases`
3. Run tests to verify catalog consistency: `pytest tests/test_integration.py::TestProductCatalog -v`

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes
3. Run tests and linting
4. Submit a PR with a clear description

## Reporting Issues

Use GitHub Issues. Include:
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
