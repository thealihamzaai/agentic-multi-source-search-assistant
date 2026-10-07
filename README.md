# Agentic Multi-Source Search Assistant

A lightweight AI search assistant that routes user questions to the most suitable source — **DuckDuckGo**, **Wikipedia**, or **arXiv** — and uses a Groq-hosted LLM to synthesize a concise answer.

## What it demonstrates

- LLM-based tool routing
- Web search integration
- Wikipedia lookup
- arXiv research-paper search
- Groq LLM integration
- Streamlit chat interface
- Clear separation between retrieval and answer synthesis

## How it works

1. The user asks a question.
2. The LLM selects the most appropriate source type.
3. The app retrieves information from that source.
4. The LLM synthesizes the retrieved material into a concise response.
5. The interface shows which source type and search query were used.

## Tech stack

- Python
- Streamlit
- LangChain Community tools
- Groq
- DuckDuckGo Search
- Wikipedia
- arXiv

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file from `.env.example` and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Run the app:

```bash
streamlit run app.py
```

## Example questions

- What are the latest developments in multimodal AI?
- Give me background on the transformer architecture.
- Find recent arXiv research on retrieval-augmented generation.

## Security

- Never commit `.env` or API keys.
- The repository includes `.env.example` only.
- Rotate any API key that may previously have been stored in a shared or committed `.env` file.

## Limitations

- Search quality depends on the selected external source.
- Web results can change over time.
- The app currently routes each question to one primary source rather than combining all three.
- LLM-generated summaries should be checked when used for high-stakes decisions.

## Project status

Portfolio/demo project focused on tool-using AI workflows and information retrieval.
