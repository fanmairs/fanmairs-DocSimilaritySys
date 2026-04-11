from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"
FRONTEND_INDEX_FILE = FRONTEND_DIST_DIR / "index.html"


def serve_frontend_path(request_path: str = ""):
    if not FRONTEND_INDEX_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail="Frontend build not found. Run `npm run build` in the frontend directory first.",
        )

    if request_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")

    if not request_path:
        return FileResponse(FRONTEND_INDEX_FILE)

    candidate = (FRONTEND_DIST_DIR / request_path).resolve()
    try:
        candidate.relative_to(FRONTEND_DIST_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Invalid frontend asset path") from exc

    if candidate.is_file():
        return FileResponse(candidate)

    return FileResponse(FRONTEND_INDEX_FILE)
