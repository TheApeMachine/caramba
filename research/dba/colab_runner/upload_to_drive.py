#!/usr/bin/env python3
"""
Upload DBA Benchmark Notebook to Google Drive

Creates a configured notebook and uploads it to Google Drive.
Then provides a direct link to open it in Colab.

Usage:
    python upload_to_drive.py --checkpoint-dir "/DBA/checkpoints/100k"

First time setup:
    pip install google-auth google-auth-oauthlib google-api-python-client
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Import the notebook creation function from dispatch.py
from dispatch import create_notebook_with_config


def get_drive_service():
    """Authenticate and return Google Drive service."""
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        print("Required packages not installed. Run:")
        print("  pip install google-auth google-auth-oauthlib google-api-python-client")
        sys.exit(1)

    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    creds = None
    token_path = Path.home() / '.colab_dispatcher_token.json'
    creds_path = Path(__file__).parent / 'credentials.json'

    # Load existing credentials
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    # Refresh or get new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not creds_path.exists():
                print("\n" + "="*60)
                print("GOOGLE DRIVE API SETUP REQUIRED")
                print("="*60)
                print("\nTo upload notebooks to Google Drive, you need API credentials:")
                print("\n1. Go to: https://console.cloud.google.com/apis/credentials")
                print("2. Create a new project (or select existing)")
                print("3. Enable the Google Drive API")
                print("4. Create OAuth 2.0 credentials (Desktop application)")
                print("5. Download the credentials JSON file")
                print(f"6. Save it as: {creds_path}")
                print("\nThen run this script again.")
                print("="*60 + "\n")
                sys.exit(1)

            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)

        # Save credentials for next run
        with open(token_path, 'w') as f:
            f.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)


def upload_notebook_to_drive(notebook_path: str, folder_name: str = "DBA_Benchmarks") -> str:
    """
    Upload notebook to Google Drive and return the file ID.

    Args:
        notebook_path: Path to the local notebook file
        folder_name: Name of the Drive folder to upload to

    Returns:
        Google Drive file ID
    """
    from googleapiclient.http import MediaFileUpload

    service = get_drive_service()

    # Find or create the folder
    folder_query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=folder_query, spaces='drive').execute()
    folders = results.get('files', [])

    if folders:
        folder_id = folders[0]['id']
    else:
        # Create folder
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = service.files().create(body=folder_metadata, fields='id').execute()
        folder_id = folder['id']
        print(f"Created Drive folder: {folder_name}")

    # Upload the notebook
    file_name = Path(notebook_path).name
    file_metadata = {
        'name': file_name,
        'parents': [folder_id],
        'mimeType': 'application/vnd.google.colaboratory'
    }

    media = MediaFileUpload(
        notebook_path,
        mimetype='application/json',
        resumable=True
    )

    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, webViewLink'
    ).execute()

    return file['id'], file.get('webViewLink', '')


def main():
    parser = argparse.ArgumentParser(
        description="Upload DBA benchmark notebook to Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Google Drive path to checkpoints (e.g., '/DBA/checkpoints/100k')"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="/DBA/results",
        help="Google Drive path for results (default: /DBA/results)"
    )

    parser.add_argument(
        "--tests-per-category",
        type=int,
        default=30,
        help="Number of tests per category (default: 30)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--webhook",
        type=str,
        default=None,
        help="Webhook URL for notifications"
    )

    parser.add_argument(
        "--github-repo",
        type=str,
        default="theapemachine/caramba",
        help="GitHub repo to clone (default: theapemachine/caramba)"
    )

    parser.add_argument(
        "--github-branch",
        type=str,
        default="main",
        help="GitHub branch (default: main)"
    )

    args = parser.parse_args()

    print("="*60)
    print("DBA Colab Notebook Uploader")
    print("="*60)

    # Normalize Drive paths
    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.startswith("/content/drive/MyDrive"):
        checkpoint_dir = f"/content/drive/MyDrive{checkpoint_dir}"

    results_dir = args.results_dir
    if not results_dir.startswith("/content/drive/MyDrive"):
        results_dir = f"/content/drive/MyDrive{results_dir}"

    print(f"\nConfiguration:")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Results: {results_dir}")
    print(f"  Tests/category: {args.tests_per_category}")

    # Create configured notebook
    print("\nCreating configured notebook...")
    notebook_path = create_notebook_with_config(
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        tests_per_category=args.tests_per_category,
        seed=args.seed,
        webhook=args.webhook,
        github_repo=args.github_repo,
        github_branch=args.github_branch,
    )
    print(f"Notebook created: {notebook_path}")

    # Upload to Drive
    print("\nUploading to Google Drive...")
    file_id, web_link = upload_notebook_to_drive(notebook_path)

    # Generate Colab URL
    colab_url = f"https://colab.research.google.com/drive/{file_id}"

    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"\nNotebook uploaded to Google Drive.")
    print(f"\nOpen in Colab:")
    print(f"  {colab_url}")
    print("\nSteps:")
    print("  1. Click the link above")
    print("  2. Go to Runtime > Change runtime type > GPU")
    print("  3. Click Runtime > Run all")
    print("  4. Authorize Google Drive access when prompted")
    print("="*60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
