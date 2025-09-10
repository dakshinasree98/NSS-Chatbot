import pandas as pd

# Load FAQ Excel only once when the module is imported
faq_df = pd.read_excel("faq.xlsx")

async def handle_faq(message_text, classification_result, phone_number):
    matched_answer = None
    for _, row in faq_df.iterrows():
        if str(row["Question"]).lower() in message_text.lower():
            matched_answer = row["Answer"]
            break

    # If no match found
    if not matched_answer:
        matched_answer = "Sorry, I couldn’t find an exact answer to your question. Our team will get back to you."

    return {
        "phone_number": phone_number,
        "ai_response": matched_answer,
        "ai_reason": classification_result["Classification"],
        "ai_classification": classification_result["Classification"],
        "ai_sub_classification": classification_result["Sub_Classification"],
    }
