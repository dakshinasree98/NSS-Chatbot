async def handle_donations(message_text, classification_result, phone_number):
    answer = f"I found the following information for your query: '{message_text}'."

    return {
        "phone_number": phone_number,
        "ai_response": answer,
        "ai_reason": classification_result["Classification"],
        "ai_classification": classification_result["Classification"],
        "ai_sub_classification": classification_result["Sub_Classification"],
    }
