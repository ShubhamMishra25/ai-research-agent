# Production-Oriented AI Research Agent

A production-grade, bounded AI Research Agent built using **Python 3.11+**, **LangGraph**, **LangChain Google GenAI (Gemini)**, and **Pydantic**.

## Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure credentials:
   ```bash
   copy .env.example .env
   ```
   Set `GEMINI_API_KEY` and `TAVILY_API_KEY` in `.env`.

3. Run the agent:
   ```bash
   python -m app.main "What are the latest developments in renewable energy?"
   ```

## Testing

Run the unit tests from the generated project directory:

```bash
pytest
```
