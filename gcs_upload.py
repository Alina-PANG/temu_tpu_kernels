#!/usr/bin/env python3
"""Optional directory uploader to Google Cloud Storage via metadata token.

Used only when the caller explicitly sets `GCS_DEST=gs://<bucket>/<prefix>`.
Continues uploading remaining files if a transient error occurs on one object
and exits non-zero if any upload failed.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

_METADATA_TOKEN_URL = (
    "http://metadata.google.internal/computeMetadata/v1/"
    "instance/service-accounts/default/token"
)
_GCS_UPLOAD_BASE_URL = "https://storage.googleapis.com/upload/storage/v1/b"
_RETRYABLE_CODES = frozenset({408, 429, 500, 502, 503, 504})


def _urlopen_retry(req: urllib.request.Request, timeout_sec: int) -> bytes:
  """Executes an HTTP request with bounded exponential backoff.

  Args:
    req: Prepared `urllib.request.Request`.
    timeout_sec: Per-attempt socket timeout in seconds.

  Returns:
    Raw response body bytes.
  """
  for attempt in range(3):
    try:
      with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return resp.read()
    except urllib.error.HTTPError as err:
      if err.code not in _RETRYABLE_CODES or attempt == 2:
        raise
    except (urllib.error.URLError, TimeoutError):
      if attempt == 2:
        raise
    time.sleep(float(1 << attempt))
  raise RuntimeError("Exhausted retries.")


def main() -> int:
  """Uploads `sys.argv[1]` recursively to `gs://...` in `sys.argv[2]`."""
  if len(sys.argv) != 3 or not sys.argv[2].startswith("gs://"):
    print(
        "Usage: gcs_upload.py <local_dir> gs://<bucket>/<prefix>",
        file=sys.stderr,
    )
    return 2
  local_dir = pathlib.Path(sys.argv[1])
  bucket, _, prefix = sys.argv[2].removeprefix("gs://").partition("/")
  prefix = prefix.strip("/")

  tok_req = urllib.request.Request(
      _METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"}
  )
  token = json.loads(_urlopen_retry(tok_req, timeout_sec=10))["access_token"]

  files = sorted(p for p in local_dir.rglob("*") if p.is_file())
  uploaded = 0
  failed = 0
  for path in files:
    rel = path.relative_to(local_dir).as_posix()
    blob_name = f"{prefix}/{rel}" if prefix else rel
    encoded = urllib.parse.quote(blob_name, safe="")
    url = f"{_GCS_UPLOAD_BASE_URL}/{bucket}/o?uploadType=media&name={encoded}"
    req = urllib.request.Request(
        url,
        data=path.read_bytes(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        },
        method="POST",
    )
    try:
      _urlopen_retry(req, timeout_sec=180)
      uploaded += 1
    except Exception as err:  # pylint: disable=broad-exception-caught
      failed += 1
      print(f"WARN failed to upload {rel}: {err}", file=sys.stderr, flush=True)

  print(
      f"Uploaded {uploaded}/{len(files)} files to gs://{bucket}/{prefix}"
      f" (failed={failed}).",
      flush=True,
  )
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main())
