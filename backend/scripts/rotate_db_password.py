"""Rotate the app's database password without it ever being printed or sent in plain text.

Generates a new password, sets it on the database as a SCRAM-SHA-256 verifier computed locally (so the server and
its query logs only ever see the hash), rewrites DATABASE_URL in backend/.env, and writes the new URL to OUT_FILE
(owner-only permissions) for updating the Railway services, e.g.:

    python scripts/rotate_db_password.py /path/to/new-url
    for s in api daily refresh; do railway variable set DATABASE_URL --stdin --service $s < /path/to/new-url >/dev/null; done
    rm /path/to/new-url
"""

import base64
import hashlib
import hmac
import os
import re
import secrets
import sys
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import psycopg
from dotenv import dotenv_values
from psycopg import sql

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
ITERATIONS = 4096


def scram_verifier(password: str) -> str:
    salt = secrets.token_bytes(16)
    salted = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    client_key = hmac.new(salted, b"Client Key", hashlib.sha256).digest()
    server_key = hmac.new(salted, b"Server Key", hashlib.sha256).digest()
    stored_key = hashlib.sha256(client_key).digest()
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return f"SCRAM-SHA-256${ITERATIONS}:{b64(salt)}${b64(stored_key)}:{b64(server_key)}"


def with_password(url: str, password: str) -> str:
    parts = urlsplit(url)
    netloc = f"{parts.username}:{quote(password, safe='')}@{parts.hostname}" + (f":{parts.port}" if parts.port else "")
    return urlunsplit(parts._replace(netloc=netloc))


def can_connect(url: str) -> bool:
    try:
        with psycopg.connect(url, connect_timeout=10) as conn:
            conn.execute("select 1")
        return True
    except psycopg.OperationalError:
        return False


def main(out_file: str) -> None:
    old_url = dotenv_values(ENV_FILE)["DATABASE_URL"]
    new_password = secrets.token_hex(24)
    new_url = with_password(old_url, new_password)

    with psycopg.connect(old_url, connect_timeout=10) as conn:
        conn.execute(sql.SQL("ALTER ROLE CURRENT_USER PASSWORD {}").format(sql.Literal(scram_verifier(new_password))))

    env = ENV_FILE.read_text()
    ENV_FILE.write_text(re.sub(r"^DATABASE_URL=.*$", lambda _: f"DATABASE_URL={new_url}", env, flags=re.M))
    fd = os.open(out_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(new_url)

    print(f"new password works: {can_connect(new_url)}")
    print(f"old password rejected: {not can_connect(old_url)}")


if __name__ == "__main__":
    main(sys.argv[1])
