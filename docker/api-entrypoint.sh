#!/bin/sh
set -eu

python - <<'PY'
import os
import sys
import time
import urllib.error
import urllib.request

host = os.environ.get("WEAVIATE_HOST", "weaviate")
port = os.environ.get("WEAVIATE_PORT", "8080")
url = f"http://{host}:{port}/v1/.well-known/ready"
deadline = time.monotonic() + float(os.environ.get("WEAVIATE_STARTUP_TIMEOUT_SECONDS", "180"))
last_error = "not attempted"

while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            if 200 <= response.status < 300:
                print(f"weaviate_ready url={url}", flush=True)
                break
            last_error = f"HTTP {response.status}"
    except (OSError, urllib.error.URLError) as error:
        last_error = str(error)
    time.sleep(3)
else:
    print(f"weaviate_not_ready url={url} reason={last_error}", file=sys.stderr, flush=True)
    raise SystemExit(1)
PY

exec "$@"
