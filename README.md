# RAG Project

This Project is focused on implementing a Retrieval-Augmented Generation (RAG) system using machine learning techniques.

The LLM used in this project is "meta/muse-glimmer-30b" and "gpt-5.4-mini". Please make sure to set the `NVIDIA_API_KEY` and `OPENAI_API_KEY` environment variables in your `.env` file.

## Requirements

This project uses [mise](https://mise.jdx.dev/) for tool version management and [uv](https://docs.astral.sh/uv/) for Python dependency management.

### Setup

1. **Install mise** (manages tool versions for this project):
   https://mise.jdx.dev/getting-started.html

2. **Install uv** (Python package and project manager):
   https://docs.astral.sh/uv/getting-started/installation/

3. **Install the project:**

```bash
   mise trust       # trusts the tool versions defined in mise.toml
   mise install     # installs the tool versions defined in mise.toml
   uv sync          # creates the venv and installs all dependencies from uv.lock
```

### Usage

- Run code inside the project environment:

```bash
  uv run python -m rag_project.llm
```

- Add a new dependency (installs it and registers it in `pyproject.toml`):

```bash
  uv add <package>
```

- Rebuild the environment after pulling changes:

```bash
  uv sync
```

No manual virtualenv activation is needed — `uv run` always uses the correct environment.

### Linting and formatting

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting. It is
installed as part of the `dev` dependency group, so `uv sync` already provides it.

```bash
  uv run ruff check .          # lint the project
  uv run ruff check --fix .    # lint and apply safe fixes
  uv run ruff format .         # format the code
```

In VS Code, the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
is configured as the default Python formatter (see `.vscode/settings.json`): files are
formatted on save and imports are organized automatically.
