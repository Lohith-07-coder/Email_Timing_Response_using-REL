import os
import time

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Assuming these are available in your project namespace
from main import get_trained_agent
from simulation.sources.nlp_extractor import NLPEmailExtractor

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def authenticate_gmail():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds)
    return service


def get_unread_emails(service):
    """Fetch unread emails from Inbox"""
    try:
        results = (
            service.users()
            .messages()
            .list(userId="me", q="is:unread in:inbox")
            .execute()
        )
        messages = results.get("messages", [])
        return messages
    except Exception as error:
        print(f"An error occurred: {error}")
        return []


def get_message_headers(service, msg_id):
    """Gets the subject and sender of the email"""
    msg = (
        service.users()
        .messages()
        .get(
            userId="me",
            id=msg_id,
            format="metadata",
            metadataHeaders=["Subject", "From"],
        )
        .execute()
    )
    headers = msg["payload"]["headers"]

    subject = "No Subject"
    sender = "Unknown Sender"

    for header in headers:
        if header["name"].lower() == "subject":
            subject = header["value"]
        if header["name"].lower() == "from":
            sender = header["value"]

    return subject, sender


def apply_action_to_gmail(service, msg_id, action_label):
    """Applies the actual action to Gmail based on the Agent's decision"""
    try:
        # Define modifications based on ML label
        if action_label == "archive":
            # Remove INBOX label to archive it
            body = {"removeLabelIds": ["INBOX"]}
            service.users().messages().modify(
                userId="me", id=msg_id, body=body
            ).execute()
            print("  -> Action: Archived email.")

        elif action_label == "mark_important":
            # Add IMPORTANT label
            body = {"addLabelIds": ["IMPORTANT"]}
            service.users().messages().modify(
                userId="me", id=msg_id, body=body
            ).execute()
            print("  -> Action: Marked as IMPORTANT.")

        elif action_label == "delay_reply":
            # Just add our own custom label or star it, or leave it in inbox
            body = {"addLabelIds": ["STARRED"]}
            service.users().messages().modify(
                userId="me", id=msg_id, body=body
            ).execute()
            print("  -> Action: Starred/Delayed reply.")

        elif action_label == "reply_now":
            # Star it and mark important, or keep in inbox
            body = {"addLabelIds": ["STARRED", "IMPORTANT"]}
            service.users().messages().modify(
                userId="me", id=msg_id, body=body
            ).execute()
            print("  -> Action: Flagged for IMMEDIATE REPLY.")

        # Finally, mark the email as read so we don't process it again next loop
        service.users().messages().modify(
            userId="me", id=msg_id, body={"removeLabelIds": ["UNREAD"]}
        ).execute()

    except Exception as error:
        print(f"An error occurred modifying message: {error}")


def run_gmail_integration():
    print("1. Authenticating to Gmail...")
    service = authenticate_gmail()

    print("2. Loading ML Agent and NLP Extractor...")
    agent = get_trained_agent(mode="dqn")  # uses existing weights logic
    extractor = NLPEmailExtractor()

    # Run a loop or just run once
    print("3. Checking for new emails...")
    messages = get_unread_emails(service)

    if not messages:
        print("No new unread messages.")
        return

    print(f"Found {len(messages)} new message(s). Processing...")

    for msg in messages:
        msg_id = msg["id"]
        subject, sender = get_message_headers(service, msg_id)

        print(f"\nProcessing Email: '{subject}' from '{sender}'")

        # Extract features using existing NLP
        email_obj = extractor.extract(subject, sender)
        state_vector = email_obj.to_state_vector()

        # Decide action!
        action_idx = agent.select_action(state_vector)
        action_label = ["reply_now", "delay_reply", "mark_important", "archive"][
            action_idx
        ]

        print(f"  -> Agent Decision: {action_label}")

        # Apply to Gmail
        apply_action_to_gmail(service, msg_id, action_label)
        time.sleep(1)  # Be gentle with API rate limits


if __name__ == "__main__":
    run_gmail_integration()
