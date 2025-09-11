# handlers/greetings.py
import re
from typing import Dict, Any, Optional

async def handle_greeting(
    message_text: Optional[str],
    user_name: str,
    classification_result: Dict[str, Any],
    phone_number: str
) -> Dict[str, Any]:
    """
    Return a response that mirrors the user's greeting and includes the user's name
    unless the greeting already contains the name.
    """

    # Helper to get classification keys robustly
    def _get_classification(key_variants, default=""):
        for k in key_variants:
            if isinstance(classification_result, dict) and k in classification_result:
                return classification_result[k]
        return default

    ai_reason = _get_classification(["Classification", "classification"], "GREETING RELATED TEXT")
    ai_classification = ai_reason
    ai_sub_classification = _get_classification(
        ["Sub_Classification", "sub_classification", "subClassification", "sub_class"], ""
    )

    # Normalize incoming text
    text = (message_text or "").strip()

    # If empty, fall back to a simple greeting
    if not text:
        greeting_core = "Hello"
    else:
        # Use first non-empty line (ignore any trailing lines)
        greeting_core = text.splitlines()[0].strip()

        # Trim leading/trailing whitespace and a few trailing punctuation characters, while preserving emojis/non-latin
        # We only strip common punctuation (.,!?;:) from the ends, but leave emojis and letters intact.
        greeting_core = re.sub(r'^[\s\.\,\!\?\;\:]+', '', greeting_core)
        greeting_core = re.sub(r'[\s\.\,\!\?\;\:]+$', '', greeting_core)

        # If stripping removed everything (e.g., message was only punctuation),
        # keep original trimmed text to preserve emojis.
        if not greeting_core:
            greeting_core = text.strip()

    # Determine whether to append user_name
    # Lowercase compare ignoring accents to be forgiving
    try:
        lower_greeting = greeting_core.lower()
        lower_name = (user_name or "").lower()
    except Exception:
        lower_greeting = greeting_core
        lower_name = user_name or ""

    if lower_name and lower_name in lower_greeting:
        # Name already present — don't append it again
        greeting_with_name = greeting_core
    else:
        # Append name with a space (avoid double spaces)
        if greeting_core:
            greeting_with_name = f"{greeting_core} {user_name}".strip()
        else:
            greeting_with_name = f"Hello {user_name}".strip()

    # Ensure punctuation at end (one exclamation or period)
    if not re.search(r'[\.\!\?]\s*$', greeting_with_name):
        greeting_with_name = greeting_with_name + "!"

    # Final friendly message
    ai_response = f"{greeting_with_name} How can I assist you today?"

    return {
        "phone_number": phone_number,
        "ai_response": ai_response,
        "ai_reason": ai_reason,
        "ai_classification": ai_classification,
        "ai_sub_classification": ai_sub_classification,
    }
