from langchain_core.tools import tool

@tool
def send_email(to_address: str, subject: str, body: str) -> str:
    """Send a mock email notification. Use this when you need to alert the user, supervisors, or the billing department.
    
    Args:
        to_address: The recipient email address.
        subject: The subject of the email.
        body: The main content/body of the email.
    """
    # Simply log the email structure mock to the console
    print(f"\n======== [MOCK EMAIL SENT] ========")
    print(f"TO:      {to_address}")
    print(f"SUBJECT: {subject}")
    print(f"BODY:\n{body}")
    print(f"====================================\n")
    return f"Success: Mock email sent to {to_address} with subject '{subject}'."
