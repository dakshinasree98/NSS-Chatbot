from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
import os
from dotenv import load_dotenv
import uvicorn
from typing import Optional, Any, Dict
from contextlib import asynccontextmanager
from loguru import logger
import sys
import asyncio
import uuid
import json
from supabase import create_client, Client
from google import genai
from google.genai.types import Part
import httpx
import requests
from handlers.greetings import handle_greeting
from handlers.faq import handle_faq
from handlers.donations import handle_donations
import re

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# Configure logger
# ----------------------------
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    colorize=True
)

if os.getenv("DEBUG", "False").lower() != "true":
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO"
    )

# ----------------------------
# Supabase client
# ----------------------------
def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    # Use service key for server-side operations to bypass RLS
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        logger.error("SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_ANON_KEY) must be set")
        raise ValueError("Missing Supabase configuration")
    
    if os.getenv("SUPABASE_SERVICE_KEY"):
        logger.info("Using Supabase service key (bypasses RLS)")
    else:
        logger.warning("Using anon key - ensure RLS policies allow inserts")
    
    return create_client(url, key)

# ----------------------------
# Gemini AI Classification Setup
# ----------------------------
def get_gemini_client():
    """Initialize Gemini client with API key from environment"""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return None
    
    try:
        client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {str(e)}")
        return None

# ----------------------------
# Image Analysis Function
# ----------------------------
async def analyze_image_with_gemini(image_url: str, gemini_client) -> dict:
    """Analyze image using Gemini API and return transcription"""
    
    if not gemini_client:
        return {
            "transcription": "",
            "status": "error",
            "error": "Gemini client not available"
        }
    
    try:
        logger.info(f"Analyzing image from URL: {image_url}")
        
        # Download image
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()
        
        # Get MIME type from response headers or default to jpeg
        mime_type = resp.headers.get('content-type', 'image/jpeg').split(';')[0]
        
        # Create Part from bytes with explicit mime_type
        image_part = Part.from_bytes(data=resp.content, mime_type=mime_type)
        
        # Send to Gemini
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Explain what is in this image clearly and in detail.",
                image_part
            ]
        )
        
        transcription = response.text.strip()
        logger.info(f"Image analysis completed. Transcription length: {len(transcription)}")
        
        return {
            "transcription": transcription,
            "status": "success",
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return {
            "transcription": "",
            "status": "error", 
            "error": str(e)
        }

# Few-shot examples for classification
FEW_SHOT_EXAMPLES = """
DONATION & TICKET RELATED ENQUIRIES:
- "I'm interested . Please mention purpose of acceptance of donation. I have send to feed Divang children in memory of my demised wife Manorama Vijay."
- "Can we send contribution in the above account?"
- "Please send receipt"
- "Pls send pay tm link for donation"
- "Kindly send my last year's donation recepit for income tax"
- "Send final receipt to claim rebate"
- "Payment karo 900"
- "6500/- paid"
- "Receipt plz ??"
- "My today's donation ☝️my donation Id is 1047733"
- "Finally I succeeded today in transferring ₹2000/- to the sanstha a/c for feeding children on amavasya.🙏"
- "Donation for bhadrapada shani amavasya"
- "For children's meal"
- "Can you share 1 pdf for all 12 months donation for tax itr"

EDUCATION & TRAINING ENQUIRIES:
- "Us time pese pareeksha aa gahi thi to me sekhne ke liye nahi aa para tha"
- "20.9.25ke bad muje computer course karna h"
- "आप के यहां विकलांगो को part time जॉब मिलेगा क्या"
- "Education jankari chahiye"

GENERAL INFORMATION ENQUIRIES:
- "Glad to have information sent by you"
- "Aapka fecebook me video dekh kar aap se sampark kar rahe hai"
- "में आप से मिलना चाहता हूँ बच्चे के साथ"
- "Katha karna chahta hu"
- "जानकारी दीजिये"
- "Narayan Seva Sansthan se. Ham judna chahte hain. aur katha bhi karna chahte hain."
- "Katha karne ke liye tyar hai"
- "Call kb kr skte he"
- "कब का डेट हो सकता है कथा हमारी"
- "M sansthan ka Sahyog karna chahta hu"
- "Sir mujhe apni bhanji ko dikhana hai"
- "Lucknow me kab tak camp lage ga"
- "कैंप kha lage ga"

GREETING RELATED TEXT:
- "Ok thanks."
- "Ok"
- "Dhanyawad Shri Radhe😊"
- "Jay -jay shri narayan, jay narayan, jay shri narayan. 🙏🙏🙏🙏🙏🙏🙏"
- "आपका बहुत बहुत आभार"
- "Hii"
- "🙏💐 Ram Ram ji 🙏🙏"
- "Jay shree shyam..."
- "Namaste👃👃"
- "Jai shree shyam 🙏"
- "Good morning saheb. God bless all of you and have a nice day."
- "Ram ram ji"
- "Jai Narayan"

MEDICAL / TREATMENT ENQUIRIES:
- "सर क्या यह सुविधा मुझे हरिद्वार में मिल सकता है हरिद्वार फिजियोथैरेपी केंद्र का डिटेल्स मिल सकता है"
- "मैं अपने बच्चों के लिए दिखाना चाहता हूं इस संस्था में कब आना होगा"
- "मैं दोनों पैरों से दिव्यांग हु"
- "I am a disabled person."
- "My name is, Randhir Kumar Singh,age-42,I am from bihar.my left leg go back hipper from knee.I feel very pain and my walking is difficult have you any solution?"
- "तो मेरे बेटे का ऑपरेशन का बोला था"
- "Above knee left leg"
- "Tumor near liver ..need to get surgery done ...robotic"
- "Kya mujhe usme above knee artificial limb mil sakta hai"
- "Sir ji mera train se pair kat gya tha"
- "मेरा नाम राजकुमार है।मेरा दाहिने पैर गुदने के निचे से कट गया है।"
- "Me ek pair se viklang"

OPERATIONAL / CALL HANDLING ENQUIRIES:
- "Please call me"
- "Not connecting your number"
- "विश्व प्रेम संस्थान. वृन्दावन आपके नारायण सेवा संस्थान की सेवा के लिए सदैव समर्पित है.... आप 9675333000 पर संपर्क कर सकते हैं"

SPAM:
- Instagram/Facebook/YouTube links
- Unrelated promotional content
- Random forwards and links
"""

def classify_message_with_gemini(message: str, gemini_client) -> dict:
    """Classify message using Gemini API with the updated FEW_SHOT_EXAMPLES"""
    
    if not gemini_client:
        logger.warning("Gemini client not available. Defaulting classification.")
        return {
            "classification": "GENERAL INFORMATION ENQUIRIES",
            "confidence": "LOW",
            "reasoning": "Gemini client not available"
        }
    
    if not message or message.strip() == "":
        return {
            "classification": "GENERAL",
            "sub_classification": "Greeting",
            "confidence": "MEDIUM",
            "reasoning": "Empty or whitespace message"
        }
    
    prompt = f"""
You are a message classification system for a social service organization. Based on the following examples, classify the given message into one of the main categories and a sub-classification.

Here are the categories and their sub-classifications with examples:

{FEW_SHOT_EXAMPLES}

Now classify this message: "{message}"

Respond in this exact JSON format, including 'sub_classification':
{{
    "classification": "CATEGORY_NAME",
    "sub_classification": "SUB_CATEGORY_NAME",
    "confidence": "HIGH/MEDIUM/LOW",
    "reasoning": "Brief explanation for the classification"
}}
"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        # Parse the JSON response
        result_text = response.text.strip()
        
        # Clean up the response if it has markdown formatting
        if result_text.startswith("```json"):
            result_text = result_text.replace("```json", "").replace("```", "").strip()
        
        result = json.loads(result_text)
        logger.info(f"Message classified as: {result.get('classification')} with confidence: {result.get('confidence')}")
        return result
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error from Gemini response: {e}")
        return {
            "classification": "GENERAL INFORMATION ENQUIRIES",
            "sub_classification": "Unknown",
            "confidence": "LOW",
            "reasoning": f"JSON parsing error: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return {
            "classification": "GENERAL INFORMATION ENQUIRIES",
            "sub_classification": "Unknown",
            "confidence": "LOW",
            "reasoning": f"Gemini API error: {str(e)}"
        }

supabase: Client = None
gemini_client = None

# ----------------------------
# Lifespan handler
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase, gemini_client
    logger.info("Starting FastAPI app on port {}", os.getenv('PORT', 10000))
    
    try:
        supabase = get_supabase_client()
        logger.info("Supabase Configuration: ✓ Set")
    except Exception as e:
        logger.error("Supabase connection failed: {}", e)
    
    try:
        gemini_client = get_gemini_client()
        if gemini_client:
            logger.info("Gemini AI Configuration: ✓ Set")
        else:
            logger.warning("Gemini AI Configuration: ✗ Not available")
    except Exception as e:
        logger.error("Gemini AI initialization failed: {}", e)
    
    yield
    logger.info("Application shutdown complete")

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(
    title="WhatsApp Message Processor with AI Classification",
    description="WhatsApp message processing service with AI classification and Supabase logging",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Helper functions
# ----------------------------
def serialize_datetime_recursive(obj: Any) -> Any:
    """Recursively serialize datetime objects to ISO format strings"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_datetime_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime_recursive(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_datetime_recursive(item) for item in obj)
    else:
        return obj

def to_iso(dt: Optional[datetime]) -> Optional[str]:
    """Safely convert datetime to ISO string"""
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt  # Already a string
    if isinstance(dt, datetime):
        return dt.isoformat()
    # Handle other datetime-like objects
    try:
        return dt.isoformat()
    except AttributeError:
        logger.warning(f"Cannot convert {type(dt)} to ISO format: {dt}")
        return str(dt)

# ----------------------------
# Pydantic models
# ----------------------------
class MessageRequest(BaseModel):
    WA_Auto_Id: Optional[int] = None
    WA_In_Out: Optional[str] = None
    Account_Code: Optional[int] = None
    WA_Received_At: Optional[datetime] = None
    NGCode: Optional[int] = None
    Wa_Name: Optional[str] = None
    MobileNo: Optional[str] = None
    WA_Msg_To: Optional[str] = None
    WA_Msg_Text: Optional[str] = None
    WA_Msg_Type: Optional[str] = None
    Integration_Type: Optional[str] = None
    WA_Message_Id: Optional[str] = None
    WA_Url: Optional[str] = None
    Status: Optional[str] = "success"
    Donor_Name: Optional[str] = None

class MessageResponse(BaseModel):
    phone_number: str
    ai_response: str
    ai_reason: str
    WA_Auto_Id: Optional[int] = None
    WA_Message_Id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str
    version: str

class MessageCompletenessResponse(BaseModel):
    completeness: str # "full" or "partial"
    reasoning: Optional[str] = None

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

# ----------------------------
# Improved Supabase logging
# ----------------------------
async def log_to_supabase(log_data: dict, table: str = "message_logs"):
    """Log data to Supabase with proper datetime serialization"""
    try:
        if supabase:
            # Recursively serialize all datetime objects
            serialized_data = serialize_datetime_recursive(log_data.copy())
            
            # Verify JSON serialization works (optional safety check)
            json.dumps(serialized_data)  # This will raise an exception if not serializable
            
            # Insert into Supabase
            result = supabase.table(table).insert(serialized_data).execute()
            
            logger.debug(f"Successfully logged to Supabase table '{table}' with {len(serialized_data)} fields")
            return result
            
    except (TypeError, ValueError) as e:
        # Handle JSON serialization errors
        logger.error(f"JSON serialization error: {e}")
        logger.error(f"Problematic data keys: {list(log_data.keys())}")
    except Exception as e:
        logger.error(f"Supabase log failed: {e}")
        # Don't log the full data in error to avoid sensitive info in logs
        logger.error(f"Failed to log data with {len(log_data)} fields")

# ----------------------------
# /message Forward to system 2 for testing
# ----------------------------
async def forward_message_to_replica(payload: dict):
    """Forward message payload to external replica service"""
    replica_url = "https://fastapi-bot-rosu.onrender.com/message"
    try:
        # Serialize datetime objects before sending
        safe_payload = serialize_datetime_recursive(payload)
 
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(replica_url, json=safe_payload)
            logger.info(f"Forwarded message to replica. Status: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to forward message to replica: {e}")
 
# ----------------------------
# /message endpoint with AI classification
# ----------------------------
@app.post("/message", response_model=MessageResponse, tags=["Message Processing"])
async def handle_message(request: MessageRequest):
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    phone_number = request.MobileNo or request.WA_Msg_To or "Unknown"
    payload = request.model_dump(exclude_none=True)
    # asyncio.create_task(forward_message_to_replica(payload))
    
    log_data = {
        "request_id": request_id,
        "endpoint": "/message",
        "method": "POST",
        "status": "processing",
        "processing_start_time": start_time,
        "raw_request": payload,
        "wa_auto_id": request.WA_Auto_Id,
        "wa_in_out": request.WA_In_Out,
        "account_code": request.Account_Code,
        "wa_received_at": request.WA_Received_At,
        "ng_code": request.NGCode,
        "wa_name": request.Wa_Name,
        "mobile_no": phone_number,
        "wa_msg_to": request.WA_Msg_To,
        "wa_msg_text": request.WA_Msg_Text,
        "wa_msg_type": request.WA_Msg_Type,
        "integration_type": request.Integration_Type,
        "wa_message_id": request.WA_Message_Id,
        "wa_url": request.WA_Url,
        "status": request.Status,
        "message_length": len(request.WA_Msg_Text) if request.WA_Msg_Text else 0,
        "parameters_received": len([k for k,v in payload.items() if v is not None]),
        "includes_wa_auto_id": request.WA_Auto_Id is not None,
        "includes_wa_message_id": request.WA_Message_Id is not None,
        "donor_name" : request.Donor_Name,
        "transcription": None,  # Initialize transcription field
        "ai_classification": None,
        "ai_sub_classification": None,
        "ai_confidence": None,
        "ai_reasoning": None,
        "is_partial_message": False, # New field for partial message check
        "consolidated_message_used": False # New field if consolidation happened
    }

    classification_result = {
        "classification": "GENERAL INFORMATION ENQUIRIES",
        "sub_classification": "Unknown",
        "confidence": "LOW",
        "reasoning": "Processing error or classification not performed"
    }
    final_message_to_classify = request.WA_Msg_Text

    try:
        # --- Step 1: Check if message is partial ---
        completeness_check = await classify_message_completeness(request.WA_Msg_Text, gemini_client)
        log_data["is_partial_message"] = (completeness_check.completeness == "partial")

        # --- Step 2: Fetch history and consolidate if partial ---
        if completeness_check.completeness == "partial" and request.MobileNo and gemini_client:
            log_data["consolidated_message_used"] = True
            previous_messages = await fetch_previous_messages(request.MobileNo, request.WA_Message_Id)
            if previous_messages:
                # Combine previous messages with the current one
                # Simple join, could be more sophisticated (e.g., using a specific delimiter)
                consolidated_text = "\n".join(previous_messages) + "\n" + request.WA_Msg_Text
                final_message_to_classify = consolidated_text
                logger.info(f"Consolidated {len(previous_messages)} previous messages with current message for {phone_number}")
            else:
                logger.warning(f"Message marked as partial, but could not fetch previous messages for {phone_number}. Classifying current message only.")
                final_message_to_classify = request.WA_Msg_Text # Fallback to classifying current message
        else:
            final_message_to_classify = request.WA_Msg_Text


        # --- Step 3: Handle Images (Skip Gemini Text Classification) ---
        if request.WA_Msg_Type and request.WA_Msg_Type.lower() == "image" and request.WA_Url:
            logger.info(f"Received an image message from {phone_number}, skipping Gemini text classification.")
            # Note: Image analysis itself is commented out in the original code.
            # If image analysis is needed, uncomment and integrate `analyze_image_with_gemini` here.
            classification_result = {
                "classification": "GENERAL INFORMATION ENQUIRIES",
                "sub_classification": "Image Uploaded",
                "confidence": "LOW",
                "reasoning": "Image message skipped for text classification"
            }
            log_data["transcription"] = "Image message received (no text classification)"

        # --- Step 4: Classify the (potentially consolidated) message text ---
        elif final_message_to_classify and gemini_client:
            logger.info(f"Classifying message: {final_message_to_classify[:100]}...")
            classification_result = classify_message_with_gemini(final_message_to_classify, gemini_client)
        elif not final_message_to_classify: # Handle cases where message text is empty after consolidation/initially
             classification_result = {
                "classification": "GENERAL",
                "sub_classification": "Greeting", # Or Auto Reply, depending on context
                "confidence": "MEDIUM",
                "reasoning": "No message text to classify"
            }

        # Update log data with classification results
        log_data.update({
            "ai_classification": classification_result["classification"],
            "ai_sub_classification": classification_result["sub_classification"],
            "ai_confidence": classification_result["confidence"],
            "ai_reasoning": classification_result["reasoning"],
            "message_to_classify": final_message_to_classify # Log what was actually classified
        })


        # --- Step 5: Route to correct handler based on classification ---
        response_data = None
        handler_used = "Unknown Handler"

        if classification_result["classification"].upper() == "DONATION RELATED ENQUIRIES":
            response_data = await handle_donations(
                message_text=final_message_to_classify, # Pass the message that was classified
                classification_result=classification_result,
                phone_number=phone_number
            )
            handler_used = "handle_donations"

        elif classification_result["classification"].upper() == "GENERAL":
             if classification_result["sub_classification"].upper() == "GREETING":
                 response_data = await handle_greeting(
                    message_text=final_message_to_classify,
                    user_name=request.Wa_Name or request.Donor_Name or "Sevak",
                    classification_result=classification_result,
                    phone_number=phone_number
                )
                 handler_used = "handle_greeting"
             else: # Fallback for other 'GENERAL' sub-classes if not explicitly handled
                response_data = await handle_faq( # Default to FAQ for unhandled general cases
                    message_text=final_message_to_classify,
                    classification_result=classification_result,
                    phone_number=phone_number
                )
                handler_used = "handle_faq (fallback for GENERAL)"

        elif classification_result["classification"].upper() in [
            "GENERAL INFORMATION ENQUIRIES",
            "MEDICAL / TREATMENT ENQUIRIES",
            "EDUCATION & TRAINING ENQUIRIES", # Assuming this category exists or is handled by FAQ
            "OPERATIONAL / CALL HANDLING ENQUIRIES", # Assuming this category exists or is handled by FAQ
        ]:
            response_data = await handle_faq(
                message_text=final_message_to_classify,
                classification_result=classification_result,
                phone_number=phone_number
            )
            handler_used = "handle_faq"

        else: # Default fallback if classification doesn't match any specific handler logic
            response_data = {
            "phone_number": phone_number,
            "ai_response": "Sorry, I couldn’t understand that.",
            "ai_reason": f"Unhandled classification: {classification_result.get('classification')}",
            # Include classification details in response for debugging
            "ai_classification": classification_result.get("classification"),
            "ai_sub_classification": classification_result.get("sub_classification", "")
        }
            handler_used = "Default Fallback"


        # Ensure required fields are present in response_data
        if not response_data or "phone_number" not in response_data:
             response_data = {
                "phone_number": phone_number,
                "ai_response": "An internal error occurred. Please try again.",
                "ai_reason": "Response generation failed",
                "ai_classification": classification_result.get("classification"),
                "ai_sub_classification": classification_result.get("sub_classification", "")
             }

        # Always include WA IDs if present in the request
        if request.WA_Auto_Id is not None:
            response_data["WA_Auto_Id"] = request.WA_Auto_Id
        if request.WA_Message_Id is not None:
            response_data["WA_Message_Id"] = request.WA_Message_Id

        end_time = datetime.now()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        log_data.update({
            "status": "success",
            "processing_end_time": end_time,
            "processing_duration_ms": duration_ms,
            "response_phone_number": response_data.get("phone_number"),
            "response_ai_response": response_data.get("ai_response"),
            "response_ai_reason": response_data.get("ai_reason"),
            "response_wa_auto_id": response_data.get("WA_Auto_Id"),
            "response_wa_message_id": response_data.get("WA_Message_Id"),
            "handler_used": handler_used,
            "raw_response": response_data
        })

        # Log asynchronously
        asyncio.create_task(log_to_supabase(log_data))

        logger.info(f"Request {request_id} processed successfully in {duration_ms}ms. Classification: {classification_result['classification']} -> {classification_result['sub_classification']} | Handler: {handler_used}")
        # Return only the fields defined in MessageResponse model
        return MessageResponse(
            phone_number=response_data.get("phone_number", phone_number),
            ai_response=response_data.get("ai_response", "Error generating response."),
            ai_reason=response_data.get("ai_reason", "N/A"),
            WA_Auto_Id=response_data.get("WA_Auto_Id"),
            WA_Message_Id=response_data.get("WA_Message_Id")
        )

    except Exception as e:
        end_time = datetime.now()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        log_data.update({
            "status": "error",
            "processing_end_time": end_time,
            "processing_duration_ms": duration_ms,
            "error_type": "internal_error",
            "error_message": str(e),
            "raw_response": {"error": str(e)}
        })
        # Log asynchronously
        asyncio.create_task(log_to_supabase(log_data))
        logger.error(f"Request {request_id} failed after {duration_ms}ms: {e}")
        raise HTTPException(status_code=500, detail=str(e))















    # try:
    #     # Initialize classification variables
    #     classification_result = {
    #         "classification": "GENERAL INFORMATION ENQUIRIES",
    #         "confidence": "LOW",
    #         "reasoning": "Classification not available"
    #     }
        
    #     transcription = None
        
    #     # --- Handle Images (Skip Gemini) ---
    #     if request.WA_Msg_Type and request.WA_Msg_Type.lower() == "image" and request.WA_Url:
    #         logger.info(f"Received an image message from {phone_number}, skipping Gemini.")
        
    #         # Do NOT call analyze_image_with_gemini
    #         classification_result = {
    #             "classification": "GENERAL INFORMATION ENQUIRIES",
    #             "confidence": "LOW",
    #             "reasoning": "Image skipped – not classified by Gemini to save tokens"
    #         }

    #         # Just store a note in logs
    #         log_data["transcription"] = "Image message skipped (no Gemini call)"


        # # Check if message type is image and process it
        # if request.WA_Msg_Type and request.WA_Msg_Type.lower() == "image" and request.WA_Url:
        #     logger.info(f"Processing image message from {phone_number}")
            
        #     # Analyze image and get transcription
        #     image_analysis = await analyze_image_with_gemini(request.WA_Url, gemini_client)
            
        #     if image_analysis["status"] == "success":
        #         transcription = image_analysis["transcription"]
        #         log_data["transcription"] = transcription
                
        #         # Classify the transcription instead of the original message text
        #         if transcription and gemini_client:
        #             logger.info(f"Classifying image transcription: {transcription[:100]}...")
        #             classification_result = classify_message_with_gemini(transcription, gemini_client)
        #         else:
        #             classification_result = {
        #                 "classification": "GENERAL INFORMATION ENQUIRIES",
        #                 "confidence": "LOW",
        #                 "reasoning": "Image transcription unavailable"
        #             }
        #     else:
        #         logger.error(f"Image analysis failed: {image_analysis['error']}")
        #         log_data["transcription"] = f"Error: {image_analysis['error']}"
        #         classification_result = {
        #             "classification": "GENERAL INFORMATION ENQUIRIES",
        #             "confidence": "LOW",
        #             "reasoning": f"Image analysis failed: {image_analysis['error']}"
        #         }
        

    #     # Process text messages (existing logic)
    #     elif request.WA_Msg_Text and gemini_client:
    #         logger.info(f"Classifying text message: {request.WA_Msg_Text[:100]}...")
    #         classification_result = classify_message_with_gemini(request.WA_Msg_Text, gemini_client)
    #     elif not request.WA_Msg_Text and not transcription:
    #         classification_result = {
    #             "classification": "GREETING RELATED TEXT",
    #             "confidence": "MEDIUM", 
    #             "reasoning": "No message text or image transcription available"
    #         }
        
    #     # Add classification info to log data
    #     log_data.update({
    #         "ai_classification": classification_result["classification"],
    #         "ai_confidence": classification_result["confidence"],
    #         "ai_reasoning": classification_result["reasoning"]
    #     })


    #     # --- Route to correct handler based on classification ---
    #     if classification_result["classification"].upper() == "GREETING RELATED TEXT":
    #         response_data = await handle_greeting(
    #             message_text=request.WA_Msg_Text or "",
    #             user_name=request.Wa_Name or request.Donor_Name or "Sevak",
    #             classification_result={
    #                 "Classification": classification_result["classification"],
    #                 "Sub_Classification": classification_result.get("sub_classification", "Greeting")
    #             },
    #             phone_number=phone_number
    #         )

    #     elif classification_result["classification"].upper() == "DONATION & TICKET RELATED ENQUIRIES":
    #         response_data = await handle_donations(
    #             message_text=request.WA_Msg_Text or "",
    #             classification_result={
    #                 "Classification": classification_result["classification"],
    #                 "Sub_Classification": classification_result.get("sub_classification", "")
    #             },
    #             phone_number=phone_number
    #         )

    #     elif classification_result["classification"].upper() in [
    #         "GENERAL INFORMATION ENQUIRIES",
    #         "MEDICAL / TREATMENT ENQUIRIES",
    #         "EDUCATION & TRAINING ENQUIRIES",
    #         "OPERATIONAL / CALL HANDLING ENQUIRIES",
    #     ]:
    #         response_data = await handle_faq(
    #             message_text=request.WA_Msg_Text or "",
    #             classification_result={
    #                 "Classification": classification_result["classification"],
    #                 "Sub_Classification": classification_result.get("sub_classification", "")
    #             },
    #             phone_number=phone_number
    #         )
        
    #     else:
    #         response_data = {
    #         "phone_number": phone_number,
    #         "ai_response": "Sorry, I couldn’t understand that.",
    #         "ai_reason": classification_result["classification"],
    #         "ai_classification": classification_result["classification"],
    #         "ai_sub_classification": classification_result.get("sub_classification", "")
    #     }

    #     # Always include WA IDs if present
    #     if request.WA_Auto_Id is not None:
    #         response_data["WA_Auto_Id"] = request.WA_Auto_Id
    #     if request.WA_Message_Id is not None:
    #         response_data["WA_Message_Id"] = request.WA_Message_Id

    #     end_time = datetime.now()
    #     duration_ms = int((end_time - start_time).total_seconds() * 1000)

    #     log_data.update({
    #         "status": "success",
    #         "processing_end_time": end_time,
    #         "processing_duration_ms": duration_ms,
    #         "response_phone_number": response_data["phone_number"],
    #         "response_ai_response": response_data["ai_response"],
    #         "response_ai_reason": response_data["ai_reason"],
    #         "response_wa_auto_id": response_data.get("WA_Auto_Id"),
    #         "response_wa_message_id": response_data.get("WA_Message_Id"),
    #         "raw_response": response_data
    #     })

    #     # Log asynchronously
    #     asyncio.create_task(log_to_supabase(log_data))
        
    #     logger.info(f"Request {request_id} processed successfully in {duration_ms}ms. Classification: {classification_result['classification']}")
    #     return MessageResponse(**response_data)

    # except Exception as e:
    #     end_time = datetime.now()
    #     duration_ms = int((end_time - start_time).total_seconds() * 1000)
    #     log_data.update({
    #         "status": "error",
    #         "processing_end_time": end_time,
    #         "processing_duration_ms": duration_ms,
    #         "error_type": "internal_error",
    #         "error_message": str(e),
    #         "raw_response": {"error": str(e)}
    #     })
    #     # Log asynchronously
    #     asyncio.create_task(log_to_supabase(log_data))
    #     logger.error(f"Request {request_id} failed after {duration_ms}ms: {e}")
    #     raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Additional endpoints for classification testing
# ----------------------------
@app.get("/categories")
async def get_categories():
    """Get list of available classification categories"""
    return {
        "categories": [
            "DONATION & TICKET RELATED ENQUIRIES",
            "EDUCATION & TRAINING ENQUIRIES",
            "GENERAL INFORMATION ENQUIRIES", 
            "GREETING RELATED TEXT",
            "MEDICAL / TREATMENT ENQUIRIES",
            "OPERATIONAL / CALL HANDLING ENQUIRIES",
            "SPAM"
        ]
    }

@app.post("/classify-only", tags=["Classification"])
async def classify_only(request: dict):
    """Standalone classification endpoint for testing"""
    message = request.get("WA_Msg_Text", "")
    phone_number = request.get("MobileNo", "Unknown")
    is_partial_check_needed = request.get("check_completeness", False) # Option to test completeness check
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini AI not available")
    if is_partial_check_needed:
        completeness_result = await classify_message_completeness(message, gemini_client)
        classification_result = {
            "message_completeness": completeness_result.completeness,
            "completeness_reasoning": completeness_result.reasoning
        }
        if completeness_result.completeness == "partial" and phone_number != "Unknown":
            previous_messages = await fetch_previous_messages(phone_number)
            if previous_messages:
                consolidated_message = "\n".join(previous_messages) + "\n" + message
                classification_result["consolidated_message"] = consolidated_message
                message = consolidated_message # Classify the consolidated message
            else:
                 classification_result["consolidation_warning"] = "Could not fetch previous messages."

    result = classify_message_with_gemini(message, gemini_client)
    classification_result.update(result)

    return {
        "message": message,
        "classification": result["classification"],
        "confidence": result["confidence"],
        "reasoning": result["reasoning"]
    }

# ----------------------------
# Health & metrics
# ----------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if supabase and gemini_client else "degraded",
        timestamp=datetime.now().isoformat(),
        service="whatsapp-message-processor",
        version="1.0.1"
    )

@app.get("/metrics")
async def metrics():
    return {
        "service": "whatsapp-message-processor",
        "supabase_enabled": supabase is not None,
        "gemini_ai_enabled": gemini_client is not None,
        "timestamp": datetime.now().isoformat()
    }

# ----------------------------
# Exception handlers
# ----------------------------
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "available_endpoints": ["/message", "/health", "/metrics", "/categories", "/classify-only", "/docs", "/redoc"]}
    )

@app.exception_handler(405)
async def method_not_allowed_handler(request: Request, exc):
    return JSONResponse(
        status_code=405,
        content={"error": "Method not allowed"}
    )

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    port = int(os.getenv('PORT', 10000))
    host = "0.0.0.0"
    debug_mode = os.getenv("DEBUG", "False").lower() == "true"
    workers = int(os.getenv("WORKERS", "1"))
    uvicorn.run(
        "test:app",  # replace 'test' with your filename
        host=host,
        port=port,
        workers=workers if not debug_mode else 1,
        reload=debug_mode,
        log_level="debug" if debug_mode else "info"
    )


# Health check route (keep this if you already had one)
@app.get("/")
async def root():
    return {"status": "Chatbot is running 🚀"}

# # Main chatbot route
# @app.post("/chatbot")
# async def chatbot_endpoint(request: Request):
#     data = await request.json()

#     # Extract user input
#     message_text = data.get("WA_Msg_Text", "")
#     user_name = data.get("Wa_Name") or data.get("Donor_Name") or "Sevak"
#     phone_number = data.get("Mobile") or "Unknown"

#     # 1. Run classification
#     classification_result = classify_message(message_text)

#     # 2. Route to correct handler
#     response_data = None

#     ### --- GREETINGS FUNCTIONALITY --- ###
#     if (
#         classification_result["Classification"] == "General"
#         and classification_result["Sub_Classification"] == "Greeting"
#     ):
#         response_data = await handle_greeting(
#             message_text=message_text,
#             user_name=user_name,
#             classification_result=classification_result,
#             phone_number=phone_number
#         )

#     ### --- FAQ FUNCTIONALITY --- ###
#     elif classification_result["Classification"] in [
#         "General Information Enquiries",
#         "Medical / Treatment Enquiries",
#         "Education & Training Enquiries",
#         "Operational / Call Handling Enquiries",
#     ]:
#         response_data = await handle_faq(
#             message_text=message_text,
#             classification_result=classification_result,
#             phone_number=phone_number
#         )

#     ### --- DONATION / TICKET FUNCTIONALITY --- ###
#     elif classification_result["Classification"] == "Donation & Ticket Related Enquiries":
#         response_data = await handle_donations(
#             message_text=message_text,
#             classification_result=classification_result,
#             phone_number=phone_number
#         )

#     ### --- FALLBACK --- ###
#     else:
#         response_data = {
#             "phone_number": phone_number,
#             "ai_response": "Sorry, I couldn’t understand that.",
#             "ai_reason": "Fallback",
#             "ai_classification": classification_result.get("Classification"),
#             "ai_sub_classification": classification_result.get("Sub_Classification"),
#         }

#     return response_data


# async def classify_message_completeness(message_text: str, gemini_model) -> MessageCompletenessResponse:
#     """
#     Use Gemini to determine if a message is a complete thought or a partial one.
#     This is useful for multi-line inputs like addresses.
#     """
#     if not gemini_model:
#         logger.warning("Gemini client not available. Assuming message is 'full'.")
#         return MessageCompletenessResponse(completeness="full", reasoning="Gemini client not available")

#     if not message_text or message_text.strip() == "":
#         return MessageCompletenessResponse(completeness="full", reasoning="Empty message") # Treat empty as full (nothing to process)

#     prompt = f"""
# You are a message analysis system. Your task is to determine if the provided text represents a complete thought or a partial message that is likely part of a larger input (like a multi-line address or a sentence being typed).

# Consider the following:
# - A "full" message is a complete sentence, a standalone question, a statement, or a greeting that doesn't obviously require more context to be understood.
# - A "partial" message is a fragment that seems incomplete, often like the beginning or middle of a sentence, or a list item that's not the last one.

# Analyze the following message: "{message_text}"

# Respond ONLY with a JSON object containing two keys:
# "completeness": "full" or "partial"
# "reasoning": A brief explanation for your decision.

# Example 1:
# Message: "Hello there!"
# JSON: {{"completeness": "full", "reasoning": "It's a standard greeting."}}

# Example 2:
# Message: "123 Main Street"
# JSON: {{"completeness": "partial", "reasoning": "This looks like the first line of an address, likely followed by city and zip code."}}

# Example 3:
# Message: "I want to donate"
# JSON: {{"completeness": "partial", "reasoning": "This sentence is incomplete and likely a precursor to specifying the donation amount or purpose."}}

# Example 4:
# Message: "Thank you for your help."
# JSON: {{"completeness": "full", "reasoning": "This is a complete and polite closing statement."}}

# Now, provide the JSON for the message above.
# """

#     try:
#         response = await gemini_model.generate_content_async(prompt)
#         result_text = response.text.strip()

#         # Clean up markdown formatting
#         if result_text.startswith("```json"):
#             result_text = result_text.replace("```json", "").replace("```", "").strip()

#         result = json.loads(result_text)
#         logger.info(f"Message completeness check: {result.get('completeness')} - {result.get('reasoning')}")
#         return MessageCompletenessResponse(**result)

#     except json.JSONDecodeError as e:
#         logger.warning(f"JSON parsing error from Gemini completeness check: {e}")
#         return MessageCompletenessResponse(completeness="full", reasoning=f"JSON parsing error: {str(e)}")
#     except Exception as e:
#         logger.error(f"Gemini API error during completeness check: {e}")
#         return MessageCompletenessResponse(completeness="full", reasoning=f"Gemini API error: {str(e)}")
    

# # Placeholder for fetching previous messages
# # In a real app, this would query a database (e.g., Supabase)
# # ----------------------------
# async def fetch_previous_messages(mobile_no: str, current_message_id: Optional[str] = None, limit: int = 10) -> List[str]:
#     """
#     Fetches the last 'limit' messages for a given mobile number.
#     Returns a list of message texts.
#     NOTE: This is a placeholder. Replace with actual Supabase query.
#     """
#     if not supabase:
#         logger.warning("Supabase not initialized. Cannot fetch previous messages.")
#         return []

#     logger.info(f"Attempting to fetch last {limit} messages for {mobile_no} (excluding current message ID: {current_message_id})")
#     try:
#         query = (
#             supabase.table("message_logs")
#             .select("wa_msg_text")
#             .eq("mobile_no", mobile_no)
#             .order("wa_received_at", ascending=False) # Assuming wa_received_at is a timestamp
#             .limit(limit)
#         )
#         # If we have the current message's ID, we might want to exclude it or ensure ordering is correct
#         # For simplicity, we'll assume the 'limit' fetches messages *before* the current one if ordered by time.
#         # A more robust solution would involve timestamps or explicit message IDs.

#         response = query.execute()

#         if response.data:
#             # Extract just the message text
#             # Messages are ordered newest first, so we reverse to get them in chronological order for consolidation
#             messages = [msg["wa_msg_text"] for msg in response.data if msg["wa_msg_text"]]
#             return messages[::-1] # Reverse to get chronological order
#         else:
#             logger.info(f"No previous messages found for {mobile_no}")
#             return []
#     except Exception as e:
#         logger.error(f"Error fetching previous messages from Supabase for {mobile_no}: {e}")
#         return []
    
