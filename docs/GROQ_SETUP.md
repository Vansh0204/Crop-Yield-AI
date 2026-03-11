# Groq API key for Streamlit

Set `GROQ_API_KEY` in `.streamlit/secrets.toml` for local runs, or in the Streamlit Community Cloud project secrets for deployment.

The `llm_client.get_llm_response` helper reads this value. If the key is missing or the request fails, the app returns a short fallback message instead of raising.
