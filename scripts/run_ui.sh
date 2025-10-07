
#!/usr/bin/env bash
set -euo pipefail
# Set BOOKSENSE_API to your FastAPI URL if not local
export BOOKSENSE_API="${BOOKSENSE_API:-http://127.0.0.1:8000}"
streamlit run ui/app.py
