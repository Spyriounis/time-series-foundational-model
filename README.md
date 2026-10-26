# time-series-foundational-model

A from-scratch foundation model for zero-shot probabilistic time-series forecasting. The project covers the full pipeline:
- literature review;
- real and synthetic data;
- model and training code;
- benchmark evaluation (GIFT-Eval, fev-bench);
- open weights on Hugging Face;
- a Python library API.

## Status

**Stage 1: literature review.**

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run pre-commit install
uv run pytest
```

CI runs linting (ruff), docstring validation (numpydoc), type checking (mypy) and tests inside the Docker image:

```bash
docker build -t tsfm:latest .
docker run --rm tsfm:latest uv run pytest
```

## License

MIT. See [LICENSE](LICENSE).
