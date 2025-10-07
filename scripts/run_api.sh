
#!/usr/bin/env bash
set -euo pipefail
uvicorn api.main:app --reload
