"""Start the complete DataQualityEng app on port 8000."""

from __future__ import annotations

import os

import uvicorn


def main():
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"DataQualityEng running at http://{host}:{port}", flush=True)
    uvicorn.run("backend.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
