async def handle_greeting(message_text, user_name, classification_result, phone_number):
    custom_greeting_message = f"Hello {user_name}! How can I assist you today?"

    return {
        "phone_number": phone_number,
        "ai_response": custom_greeting_message,
        "ai_reason": classification_result["Classification"],
        "ai_classification": classification_result["Classification"],
        "ai_sub_classification": classification_result["Sub_Classification"],
    }
