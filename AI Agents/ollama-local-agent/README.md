
# Pizza Restaurant AI Assistant

This project is an AI-powered assistant that answers questions about a pizza restaurant using real customer reviews. It leverages vector search (Chroma DB), embeddings (Ollama), and a local LLM to provide contextually relevant answers.

---

## Features

- **Semantic Search:** Finds the most relevant reviews for any user question using vector embeddings.
- **Local LLM Integration:** Uses a local language model (via Ollama) for generating answers.
- **Command-Line Interface:** Simple CLI for interactive Q&A.
- **Extensible:** Easily add more reviews or swap out models.

---

## Project Structure

```
ollama-local-agent/
├── app/
│   ├── pizza_assistant.py         # LLM prompt and answer logic
│   ├── pizza_assistant_cli.py     # Command-line interface
│   ├── vector.py                  # Vector store management (Chroma DB)
├── resources/
│   └── reviews.csv                # Restaurant reviews data
├── chrome_langchain_db/           # Chroma DB persistent storage (auto-created)
├── main.py                        # Application entry point
```

---

## Setup Instructions

### 1. Prerequisites

- **Python 3.9+**
- **uv** (Python package manager): [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
- **Ollama** (for local LLM and embeddings): [https://ollama.com/](https://ollama.com/)
- **Chroma DB** and **LangChain** libraries

### 2. Install Dependencies

```sh
uv sync
```

Or if you don't have a `pyproject.toml` file yet, you can install dependencies directly:

```sh
uv add pydantic langchain-core langchain-ollama langchain-chroma pandas
```

Example `pyproject.toml` dependencies section:
```toml
[project]
dependencies = [
    "pydantic",
    "langchain-core",
    "langchain-ollama", 
    "langchain-chroma",
    "pandas",
]
```

### 3. Prepare Ollama

- Download and run Ollama.
- Pull the required models (e.g., `llama3.2` for LLM, `mxbai-embed-large` for embeddings):

```sh
ollama pull llama3.2
ollama pull mxbai-embed-large
```

### 4. Add/Update Reviews

- Place your reviews in `resources/reviews.csv` (see the provided format).

### 5. Run the Application

```sh
python main.py
```

---

## How It Works

1. **VectorStoreManager** loads reviews from CSV, creates embeddings, and stores them in Chroma DB.
2. **PizzaAssistantCLI** accepts user questions, retrieves the most relevant reviews using vector search, and passes them to the LLM.
3. **PizzaAssistant** generates a context-aware answer using the reviews and the user’s question.

---

## Notes

- The first run will create the Chroma DB and index all reviews.
- You can add more reviews to `resources/reviews.csv` and delete the `chrome_langchain_db` folder to re-index.
- All models run locally; no data leaves your machine.

---

## License

MIT License