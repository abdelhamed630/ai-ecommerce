"""Local-disk avatar storage. Only a relative URL is kept in the database."""

import os
import uuid
from typing import Optional

from core.config import settings

MAX_AVATAR_BYTES = 2 * 1024 * 1024  # 2 MB
AVATAR_URL_PREFIX = "/media/avatars/"

# mime type -> the extension WE choose (the client's filename is never used)
_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
ALLOWED_CONTENT_TYPES = set(_EXTENSIONS)


def detect_image_type(data: bytes) -> Optional[str]:
    """Identify the real format from the file's magic bytes, not its name.

    This confirms the header only; it does not fully decode the image
    (that would need Pillow, deliberately not added as a dependency).
    SVG, GIF, HTML, executables, etc. are all rejected because none match.
    """
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _avatar_dir() -> str:
    path = os.path.join(settings.MEDIA_ROOT, "avatars")
    os.makedirs(path, exist_ok=True)
    return path


def save_avatar(user_id: int, data: bytes, mime_type: str) -> str:
    """Write the file under a server-generated name; return its URL path."""
    filename = f"user{user_id}_{uuid.uuid4().hex}{_EXTENSIONS[mime_type]}"
    with open(os.path.join(_avatar_dir(), filename), "wb") as fh:
        fh.write(data)
    return f"{AVATAR_URL_PREFIX}{filename}"


def remove_avatar_file(avatar_url: Optional[str]) -> None:
    """Best-effort delete. Never raises for a missing file, and only ever
    deletes inside the avatars directory (a tampered avatar_url cannot
    make this remove an arbitrary file)."""
    if not avatar_url or not avatar_url.startswith(AVATAR_URL_PREFIX):
        return
    filename = os.path.basename(avatar_url[len(AVATAR_URL_PREFIX):])
    if not filename:
        return
    directory = os.path.realpath(_avatar_dir())
    target = os.path.realpath(os.path.join(directory, filename))
    if os.path.dirname(target) != directory:
        return
    try:
        os.remove(target)
    except FileNotFoundError:
        pass
