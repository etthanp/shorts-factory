#!/usr/bin/env python3
"""
One-time (and periodic) authorization for YouTube uploads.

Run this yourself in Terminal:
    python3 authorize_youtube.py

It opens your browser so you can log into the Google account that owns your
YouTube channel and grant upload access. Saves the result to token.json,
which pipeline.py then uses to upload videos without any further login.

Because this Google Cloud OAuth app is in "Testing" mode, the resulting
access expires after 7 days - if uploads start failing with an auth error,
just re-run this script.
"""
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

ROOT = Path(__file__).resolve().parent
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def main():
    flow = InstalledAppFlow.from_client_secrets_file(str(ROOT / "client_secret.json"), SCOPES)
    creds = flow.run_local_server(port=0)
    (ROOT / "token.json").write_text(creds.to_json())
    print("\nAuthorized. Saved token.json - pipeline.py can now upload to your channel.")


if __name__ == "__main__":
    main()
