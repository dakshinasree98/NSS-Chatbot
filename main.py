from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import uvicorn
from typing import Optional, Any, Dict, List, Union
from contextlib import asynccontextmanager
from loguru import logger
import sys
import asyncio
import uuid
import json
from supabase import create_client, Client
import google.generativeai as genai
from google.genai.types import Part
import httpx
import requests
import re
import pandas as pd


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
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "app.log")
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
def get_gemini_model() -> Optional[genai.GenerativeModel]:
    """Initialize Gemini client with API key and return a model instance"""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        logger.info("Gemini model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {str(e)}")
        return None

faq_df = pd.read_excel("faq.xlsx")

from difflib import get_close_matches

async def handle_faq(message: str) -> str:
    questions = faq_df["Question"].tolist()
    matches = get_close_matches(message, questions, n=1, cutoff=0.5)
    if matches:
        answer = faq_df.loc[faq_df["Question"] == matches[0], "Answer"].values[0]
        return answer
    else:
        return "Sorry, I couldn't find an answer for your question. Could you rephrase your question?"


# ----------------------------
# Image Analysis Function
# ----------------------------
async def analyze_image_with_gemini(image_url: str,  gemini_model: genai.GenerativeModel) -> dict:
    """Analyze image using Gemini API and return transcription"""
    
    if not gemini_model:
        return {
            "transcription": "",
            "status": "error",
            "error": "Gemini model not available"
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
        response = await gemini_model.generate_content_async([
            "Describe about the image in one line",
            image_part
        ])
        
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
Classification:- Donation Related Enquiries, Sub_Classification:- Announce Related
-दिव्यांग बच्चों के भोजन हेतु सहयोग 2100/- का समर्पित करता हूँ 🙏
-5000/- बच्चों के भोजन के लिए अर्पित
-I want to contribute for today’s amavasya bhojan seva.
-Announcing donation for child marriage event.
-I have pledged 10,000 for disabled girl marriage.
-Today I am donating ₹2500 for food seva.
-कल मैंने 2000 रुपए भोजन सेवा के लिए दिए हैं 🙏
-We want to dedicate donation for girl’s wedding.
-I hereby contribute ₹5100 for differently abled children.
-दिव्यांग बच्चों के भोजन लिए सहयोग 3000 का
5000/ का
दिव्यांग कन्या विवाह  हेतु सादर समर्पित...धन्यवाद..
में संस्थान से जुड़कर करना चाहता हु।
-I will pay 4.5k

Classification:- Donation Related Enquiries, Sub_Classification:- Receipts Related
-Receipt plz??
-Please send 80G certificate for my donation.
-Can you send me my donation receipt for FY 2024-25?
-रसीद भेजना 🙏
-I haven’t received acknowledgement for my ₹5000 donation yet.
-Need scanned copy of my donation receipt.
-Missed getting 80G form, please resend.
-Mujhe apne donation ki receipt chahiye.
-Thanks. No receipt/80G benefits needed.
-rasid NAme : Shah Hansaben Manharlal
-I didn’t get receipt for 4500
-No need to send receipt pls 🙏🏻
-Yes  only send me the donation receipt for ten thousand also send hard copy by post
-Rasid Sohan Ram Prajapat ke Name se Mil jayega kya
-Recipt भेज दो na sir ji
-Please send receipt
-Sorry, actually I need the receipts for July 24 & August 24..Kindly do the needful 🙏🏻🙏🏻
-Is there any Receipt ??
-Please send receipt
-कृपया सितंबर 2024 में दी गई डोनेशन राशि रुपए 10000 की रसीद एवं इनकम टैक्स सर्टिफिकेट प्रदान करने की कृपा करें
-Can you send for final year 2024-2025
-Can you please share all for last financial year
-U can share the receipt..if possible??
-Nd send the receipt again with correct name..
-Thanks. No receipt/80G benefits needed.
-rasid NAme : Shah Hansaben Manharlal
-"Pls. Send receipt  of deposit amount
-रसीद की हार्ड कॉपी जरूर भेजना।
-रसीद की हार्ड कॉपी जरूर भेजना।
-Subject: Request for Acknowledgement Receipts – July & August 2025
Dear Sir/Ma’am,
I have not yet received the acknowledgement receipts for the months of July 2025 and August 2025. May I kindly request you to share the same at the earliest.
Your support in this matter will be highly appreciated.
Thanks & regards,
Nilesh Bhagat 🙏🏻"
-Receipt plz ??
-Rasid. Sanjeev Kumar
-"PLEASE SEND ME RECEIPT ON WHATSAPP
-NO NEED TO SEND BY POST"

Classification:- Donation Related Enquiries, Sub_Classification:- Amount Confirmation
-500/- done from GPay.
-Transferred ₹21000, please confirm.
-यह 1500/- अभी भेजा है बच्चों के खाने के लिए
-I donated via UPI yesterday. Kindly confirm.
-Donation of 2000 sent from SBI.
-Just paid 1000 through Paytm, did you receive?
-Please confirm if 5100 has reached your account.
-Bank statement shows debit, kindly check on your side.
-Manju Agarwal 
-W/O Shri Ashok Kumar Singhal 
-R/O 6/3 A Gali barah bhai belanganj Agra"
-Ye 3000 jod jod ker mai banwa dungi
-"(New) Ms. Monika Gupta - ₹21000
Mrs. Raj Kumari Gupta - ₹9000"
Kindly acknowledge the amount I hv donated to your sanstha
-Jo Screen Shot Send kiye hai Maine
-Hi, you have sent Rs.5,500.00 from 9352351847@idfcfirst to paytmqr2810050501010uwohbemahg0@paytm using your IDFC FIRST Bank UPI.Txn ID-523676286360.
-We sent the amount for Haldi and Mehndi for two couples.
-Finally I succeeded today in transferring ₹2000/- to the sanstha a/c for feeding children on amavasya.🙏

Classification:- Donation Related Enquiries, Sub_Classification:- Donation Payment Information
-Please share account details for donation.
-QR code भेज दीजिये
-Can we donate via Paytm link?
-Which account should I use for transfer?
-Send IFSC code.
-Is credit card accepted for donation?
-Please share UPI ID.
-Kya PhonePe number hai jisme donation bhej sakte?
-Can we send contribution in the above account?
-Can we send contribution in the above account with Bank of India?
-Pls send pay tm link for donation
-There are two different account numbers. Which account should I transfer
-Pls send donation details ifsc code account
-Pl send P N  B  Actt No
-ke nama se karna he
-State bank of india
-Mam plz barcode bhej dijiye
-I am requesting you to send Bank details or QR for donations.
-Qr code beje
-Send scanner
-Please send your account details for donations

Classification:- Donation Related Enquiries, Sub_Classification:- KYC Update

-PAN no. Already shared.
-"उपरोक्त Donation मेरे द्वारा मेरे नाम Rajendra Kumar Sharma से किया गया था 
-मेरा ही Pan नम्बर दिया गया था 
-कृपया गलती सुधारने की कोशिश कीजिए 
-अन्यथा Pan नम्बर बदलने का कष्ट करें 
-This is my PAN"
-Adhar & pan
-My PAN NO is ACIPR 0141F
-"My PAN. is
-AFSPA 3996 C"
-My PAN is
-PAN NO.
-My above pan no is. Correct.
-Receiptisto inthe name of GaneshiLal yadav. Pan is also in same name. Itis already registered with you.
-Pan number AVKPS1316G
-"PAN NO
-AIFPC3542E"
-Pan no AGVPS9012M
-"AMBPK4143P
-PAN NUMBER"
-Kindly send my last year's donation recepit for income tax
-For getting 80G benefit for last year
-इस में पूरा एड्रेस लिखा हुआ है


Classification:- General, Sub_Classification:- Greeting
-Ram Ram ji 🙏
-Good morning
-Jai Shree Krishna 🌸
-🙏 Jai Narayan
-नमस्ते
-Hello
-Good evening
-Sup? How are you?

Classification:- General, Sub_Classification:- Thanks
-Thank you for the information 🙏
-धन्यवाद
-Thanks a lot
-Bahut bahut aabhar
-App ka dhanyawad
-Grateful for your support
-Shukriya 🙏
-Many thanks for clarifying
-Om Gajananaya namah. Om Mitraye namah. Radhe Radhe. Jai Sada Shiv. Jai Sarvamangala Mata. Jai Ragya Mata. Jai Bhadrakaali Mata. Jai Sharada Mata. Jai Annapurna Mata. Jai Sheetla Mata. Jai Bhoomi Mata. Jai Mangalmurti Hanuman. Om Swami Nathishoraye namah. Guru kripa. Mangalamay Mangalvaar. Orzu🙏🙏🙏
"🙏🌼🌼 *जय श्री राधाकृष्ण*🌼🌼🙏
  🙏🌺🌺 *श्रीकृष्णावतार*🌺🌺🙏

-*फिर भगवान् से मांगने की बजाये निकटता बनाओ तो सब कुछ अपने आप मिलना शुरू हो जायेगा ।*

-*🌷 जय श्री गणेश जी जय श्री कृष्ण 🌷*शुभ रात्रि  जय सियाराम"
-Jay Shri Ram
-Ram ram ji
-राधे राधे 🏵️🙏
-Gud Nyt Yu Nd Yr Family Members
-Jai naryana
-जय श्री श्याम जी 🙏
-Hi
-🙏 OK   Jay shree Radhey Krishna
"-🙏🌹🥭🪔🥥🕉️🇮🇳🙇‍♂️
-Namah vishnu in service of needy"
-Jai Narayan 👏
-जय नारायण 🙏🙏
-🙏🙏🙏 राधे राधे जय श्री कृष्ण
-हम आपके संस्थान से 2010से जुड़े हैं 🙏
-Ram ramji
-"जय नारायण 
-Jay shree Krishna 🙏
-Jai jai shree shyam 🙏🌹🙌
-Jai Shree Shyam
-Jai Shree Bala ji
Good morning sir, ji 🙏🌹
-राम राम जी  जय श्री कृष्णा जय नारायणन  हरि विष्णु जी  ।गुरूजी को चरणस्पर्श पणाम स्वीकार हो  ।

Classification:- General, Sub_Classification:- Follow-up
-Please confirm me  Narayan sevasanthan
-Jabab do
-Please
-Batao
-40 din ho gya sir
-Sir kya huaa
-Please confirm by tomorrow morning.
-"Mehandi.rashma.ki.rashi..chaqe.se.santhan.ki.khata.me.jama.karna.he
-Khaya.no.bataye"
-बताना जी
-Bahut asha hai
-Pls talk to me my phone is silent i am on line
-Any update?
-Kya hua?
-Please confirm
-40 din ho gya jawab nahi aaya
-Still waiting for reply
-Kindly update the status
-Meri request ka kya hua?
-Please get back soon

Classification:- General, Sub_Classification:- Emoji
-🙏🏻
-🌹🙏🌹
-👏👏👏
-👍👍
-❤️❤️
-👏

Classification:- General, Sub_Classification:- Interested
-I'm interested

Classification:- General, Sub_Classification:- Thanks

-Thanks Sir
-आप का बहुत धन्यवाद सेवाओं की जानकारी देने के लिए 🙏
-जी 💐 धन्यवाद 👏🏻
-Thanks
-"Apka bahut bahut dhanyavad
-🙏🙏🙏"
-Thankyou
-धन्यवाद जी जय नारायण
-धन्यवाद महोदय
-Ram ram welcome Ganeshj
-Thankyou sir

Classification:- General Information Enquiries, Sub_Classification:- About Sansthan
-What is Narayan Seva Sansthan?
-आपके संस्थान की जानकारी दीजिए
-Can you tell me about your NGO?
-I want to know about your organisation.
-Please share details about Narayan Seva.
-How does your organisation help people?
-Sansthan ki activities kya hai?
-Kya yeh trust registered hai?
आप हमें जानकारी दीजिए
hr@narayanseva.org
Seva sansthan
जानकारी दीजिये

Classification:- General Information Enquiries, Sub_Classification:- Katha Related
-Katha karna chahta hu
-नारायण सेवा संस्थान के माध्यम से यदि कोई आयोजन हो तो बताएगा।कथा के लिए
-Narayan Seva Sansthan se. Ham judna chahte hain. aur katha bhi karna chahte hain.  iske liye hamen aap sampurn jankari pradan Karen.
-Katha karne ke liye tyar hai
-कब का डेट हो सकता है कथा हमारी
-जय श्री राम नारायण सेवा संस्थान उदयपुर आपका हार्दिक अभिनंदन कभी आप एक मौका दीजिए कथा करने के लिए जय श्री राम

Classification:- General Information Enquiries, Sub_Classification:- Enquiry Visit Related

-We will arrive to UDAIPUR on 30 August in morning ,train arrival time is 8AM.Because we are coming first time to Santha so please confirm me.
-PNR:2339554510,TRN:20473,DOJ:29-08-25,SCH DEP:19:40,3A,DEE-UDZ,SANJAY KR GUPTA+1,B4 17 ,B4 20 ,Fare:002140,Please carry physical ticket. IR-CRIS
-"3 sal ki hai 
-Jila-kanpur nagar uttar pardesh 
-Kripya krke hme koi date de de angle month ki jisse ham wha tym se ake apke sanshthan me dikha sake"
-समय निकालकर सस्ता में भी आने की कोसिस करेंगे
-Ana kaha par h ye bata do aap
-UTR no 388309480581


Classification:- General Information Enquiries, Sub_Classification:- Divyang Vivah Couple

-अभी शादी कब है
-Send marriage programs card send


Classification:- General Information Enquiries, Sub_Classification:- Camp Related
-Shani amavasya ka kya hai
-Mumbai mein aapka camp kahan hai
-सर वितरण कब तक होगा
-लखनऊ में कैंप कब लगेगा?
-Is there any medical camp in September?
-Where will the next camp be organised?
-कैम्प की जानकारी दीजिए
-Please send schedule of upcoming camps.
-Any free medical camp in Delhi this month?
-Kya Jaipur mein bhi camp hota hai?


Classification:- General Information Enquiries, Sub_Classification:- Tax Related
-Kindly send my last year's donation recepit for income tax
-For getting 80G benefit for last year
-I need form for income tax filling for the donation I have done earliers


Classification:- General Information Enquiries, Sub_Classification:- School Enquiry Related

-Us time pese pareeksha aa gahi thi to me sekhne ke liye nahi aa para tha
-20.9.25ke bad muje computer course karna h


Classification:- General Information Enquiries, Sub_Classification:- Job Related
-आपके यहां नौकरी मिलेगी क्या?
-Do you have job opportunities?
-I’m looking for part time work, please guide.
-Any openings for disabled persons?
-Can I apply for work?
-Is there volunteer opportunity?
-Mujhe job ki requirement hai
-Internship provide karte ho kya?
-आप के यहां विकलांगो को part time जॉब मिलेगा क्या


Classification:- General Information Enquiries, Sub_Classification:- Financial Help

-I whant money for work then I can have food for me.


Classification:- Medical / Treatment Enquiries, Sub_Classification:- HospitalEnquiry Related

-"I had gone under CARARACT surgery for my Right eye 2 times on 16th and on 23rd of this month due to some technicap problem in my R eye. Hence I am observing screen seeing of mobile, lap top and TV.
-Please don't send any messages for coming 10 days."
-Sir. Can I get him examined at the hospital on Sunday?
-हा, hospital का शुभारंभ कब होगा????
-Apke yaha dikhana hai
-Docter ko dikhaya hai to docter bole therapy kraao or dwa pilaooo nashe jo hai tight hai
-"NEW ADDRESS 
MURARI  LAL  AGRAWAL  
-1802 A WING MODI SPACES GANGES BUILDING  OPPOSITE  BHAGWATI  HOSPITAL  BORIVALI  WEST  MUMBAI  400103"


Classification:- Medical / Treatment Enquiries, Sub_Classification:- Artificial Limb Related
-Pair bana ki nahi sir
-मैं एक विकलांग व्यक्ति हूं मेरा पैर नहीं है
-Left leg m hai sir ...
-Sir mujhe one leg m polio h mujhe thik karana hai
-A person name Raja has lost his left arm in an accident  how he can get a artificial arm...
-Ujjain se चिमनगंज थाने के आगे वाली झुग्गी झोपड़ी मे रहते हैं और ये मेरा लड़का है जिसका पैर कट गया था एक्सीडेंट में और मे ठेला लगाती हूं फ्रूट का छोटा सा
"🌈🎺🎊🥁जय श्री महाँकाल
*त्रिलोकेशं नीलकण्ठं*
           *गंगाधरं सदाशिवम् ।*
*मृत्युञ्जयं महादेवं*
           *नमामि  तं  शंकरम् ।।*
🌈भावार्थ: तीनों लोकों के स्वामी, नीलकण्ठ, गंगा को धारण करने वाले, हमेशा कल्याण करने वाले, मृत्यु पर विजय प्राप्त करने वाले, महादेव - शंकर जी की वंदना करता हूॅं।

🥁🎊🎺🌈द्वादश ज्योतिर्लिंग में तीसरे उज्जैन स्थित दक्षिणमुखी स्वयम्भू बाबा महाँकाल का आज प्रातः 4 बजे प्रारम्भ भस्म आरती श्रंगार दर्शन - 25 अगस्त 2025 शिव प्रिय सोमवार"
"पंडित श्री संतोष शास्त्री अनपूर्णा गऊ शाला शिव शक्ति खाटू श्याम बाबा परिवार नर्मदा तट मंडला वाले।
-Mo. 9753020200,7999867569
-Janm se hi biklang hai
-I lost my left leg, can I get artificial limb?
-मेरे बेटे का पैर कट गया है, क्या इलाज संभव है?
-Need artificial arm after accident.
-Do you provide prosthetic support?
-Pair bana sakte ho kya?
-Looking for artificial leg fitting.
-Mujhe hath ka prosthetic chahiye.
-Can you arrange artificial limb for my brother?

Classification:- Ticket Related Enquiry, Sub_Classification:- Receipts Related
-Please send ticket receipt.
-I have not received receipt for my ticket payment.
-रसीद अभी तक नहीं मिली
-Need acknowledgement for ticket booking.
-Can you share ticket confirmation?
-Missed receiving my ticket receipt, resend please.
-Is ticket invoice available online?
-Ticket payment done, please issue receipt.

Classification:- Spam, Sub_Classification:- YouTube / Instagram Link

Classification:- Spam, Sub_Classification:- Spammy Message

"""


async def classify_message_with_gemini(message: str, gemini_model: genai.GenerativeModel | None) -> Dict[str, Any]:
    """
    Classify message using Gemini API, extracting classification, sub-classification,
, language, script, confidence, and reasoning.
    """
    
    if not gemini_model:
        logger.warning("Gemini model not available. Defaulting classification.")
        return {
            "classification": "General Information Enquiries|Unknown",
            "Question_Language": "en",
            "Question_Script": "Latin",
            "confidence": "LOW",
            "reasoning": "Gemini model not available"
        }
    
    if not message or message.strip() == "":
        return {
            "classification": "General|Greeting",
            "Question_Language": "en", 
            "Question_Script": "Latin",
            "confidence": "MEDIUM",
            "reasoning": "Empty or whitespace message"
        }
    
    prompt = f"""
You are a message classification system for Narayan Seva Sansthan. Your primary task is to analyze the user's input and return a single, valid JSON object with:

1. "classification": "MainCategory|SubCategory"
   - MainCategory must be one from: Donation Related Enquiries, General, General Information Enquiries, Medical / Treatment Enquiries, Spam, Ticket Related Enquiry
   - SubCategory must be one from the relevant list provided below
   - Do NOT leave SubCategory as Unknown if the message clearly matches a sub-category.

Analyze the following input:
    - user Message:- {message}
    - Here are some Few Shot Examples : {FEW_SHOT_EXAMPLES}

Based on the following examples, classify the given message into one of these categories,Return a JSON object with the following schema::

"Classification": Choose the best fit from the list below, considering the conversation history for context.
        -Donation Related Enquiries
        -General
        -General Information Enquiries
        -Medical / Treatment Enquiries
        -Spam
        -Ticket Related Enquiry

"Sub_Classification": Based on the "Classification", choose one from the relevant list along with explanation:
        Donation Related Enquiries,	Announce Related,	When a donor wants to make a donation, related announcements.
        Donation Related Enquiries,	Post-Donation Related, 	When a donor shares the donation amount, details are required after deposit.
        Donation Related Enquiries,	Amount Confirmation,	To confirm whether the received amount is correctly recorded by the organization.
        Donation Related Enquiries,	Donation Payment Information,	Information required before a donor makes a donation.
        Donation Related Enquiries,	KYC Update,	After donation, KYC details are sent for updating receipts.
        Donation Related Enquiries,	Receipts Related,	Sending receipt details to donors after donation.
        Donation Related Enquiries,	Send Sadhak Related,	When a donor wants to send donation via a sadhak, including address details.
        Donation Related Enquiries,	Property Donation,	When a donor wants to donate property to the organization.
        Donation Related Enquiries,	FD & Will Related,	When a donor wants to donate FD or Will in the organization’s name.
        Donation Related Enquiries,	CSR Donation Interest,	When a company or donor wants to make a CSR donation.
        Donation Related Enquiries,	Non-Monetary Donation,	When a donor wants to donate materials instead of money.
        General, Emoji,	When a number, image, or emoji is received unrelated to the organization.
        General, Follow-up,	When a patient or donor follows up for confirmation on a previous message.
        General, Greeting,	Greetings-related messages received.
        General, Interested,	When someone expresses interest ("I'm interested") in the services or donation.
        General, Thanks,	When someone sends thanks or welcome messages.
        General, Auto Reply,	Automatic replies sent in response to broadcast messages.
        General, Ok 
        General Information Enquiries,	Suggestion,	When a donor or patient shares suggestions regarding donations.
        General Information Enquiries,	About Sansthan,	Basic information about the organization.
        General Information Enquiries,	Camp Related,	When someone requests camp-related information.
        General Information Enquiries,	Divyang Vivah Couple,	When messages are sent regarding marriage of specially-abled couples.
        General Information Enquiries,	Donation Purpose Related Information,	When someone asks about the purpose of donations or events.
        General Information Enquiries,	Enquiry Visit Related,	When a donor inquires about visiting the organization or patient stay.
        General Information Enquiries,	Financial Help,	When someone asks for financial support from the organization.
        General Information Enquiries,	Invitation Card Required,	When an event invitation card is requested.
        General Information Enquiries,	Job Related,	When someone asks about job opportunities in the organization.
        General Information Enquiries,	Katha Related,	Messages related to requesting Katha information.
        General Information Enquiries,	Woh Related,	Information about the World of Humanity new hospital.
        General Information Enquiries,	Event Related,	When a patient or donor inquires about an event.
        General Information Enquiries,	Tax Related,	Messages regarding income tax and donations.
        General Information Enquiries,	School Enquiry Related,	Admission inquiries for NCA or affiliated schools.
        General Information Enquiries,	Naturopathy Related,	Messages regarding naturopathy services.
        General Information Enquiries,	Orphanage Related Query	Messages about orphanage services or children care.
        General Information Enquiries,	Management Contact Details Required	Messages requesting management contact details.
        General Information Enquiries,	Ashram / Physiotherapy Information,	Messages about ashram or physiotherapy center information.
        General Information Enquiries,	Transportation Help Required,	Messages requesting transportation help for donors or patients.
        Medical / Treatment Enquiries,	Artificial Limb Related,	Patient inquiries related to artificial limbs.
        Medical / Treatment Enquiries,	Hospital Enquiry Related,	Hospital-related information requests.
        Medical / Treatment Enquiries,	Aids & Appliances Related,	Requests related to medical aids and appliances.
        Ticket Related Enquiry,	KYC,	After receipt completion, updating donor KYC information.
        Ticket Related Enquiry,	Master Update ,Updating donor profile like address, name, number, email, etc.
        Ticket Related Enquiry,	Receipts Related,	When a donor receipt is created but donor has not received a hard copy.
        Ticket Related Enquiry,	Complaint Related,	Donor or patient complaints regarding services or donations.
        Ticket Related Enquiry,	Beneficiaries Detail Required,	When patient list is required after a donor’s contribution.
        Ticket Related Enquiry,	Physiotherapy Center Open,	When a donor inquires about opening a physiotherapy center.
        Ticket Related Enquiry,	Receipt Book Related,	When a branch member requests receipt book details.
        Ticket Related Enquiry,	Vocational Course Related,	When a student requests information on vocational courses.
        Ticket Related Enquiry,	Branch Membership Request,	When a donor wants to become a branch member.
        Ticket Related Enquiry,	Camp Related,	When a donor wants to organize a camp through the organization.
        Ticket Related Enquiry,	Katha Related,	When someone wants to organize Katha through the organization.
        Ticket Related Enquiry,	Amount Refund Related,	Messages about double payments or refund requests.
        Ticket Related Enquiry,	Sansthan Documents Request,	Requests for organization-related documents for promotion purposes.
        Ticket Related Enquiry,	Bhojan Seva Sponsorship,	When a donor wants to sponsor Bhojan Seva.
        Ticket Related Enquiry,	CSR Document Required,	When a company requests CSR-related documents.
        Spam	Spammy Message	Messages unrelated to the organization, considered spam.
        Spam	YouTube / Instagram Link	Links shared that are not related to the organization.


ALWAYS strictly return a JSON object in the exact following format:
{{
    "classification": "MainCategory|SubCategory",
    "Question_Language": "ISO language code, e.g., en, hi",
    "Question_Script": "Script name, e.g., Latin, Devanagari",
    "confidence": "HIGH | MEDIUM | LOW",
    "reasoning": "A short one-line explanation for your classification."
}}

"""

    try:
        response = await gemini_model.generate_content_async([prompt]) 
        
        # Parse the JSON response
        result_text = response.text.strip()
        
        # Clean up the response if it has markdown formatting
        if result_text.startswith("```json"):
            result_text = result_text.replace("```json", "").replace("```", "").strip()
        
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing error from Gemini response: {e}. Response text: {result_text}")
            return {
                "classification": "General Information Enquiries|Unknown",
                "Question_Language": "en", 
                "Question_Script": "Latin",
                "confidence": "LOW",
                "reasoning": f"JSON parsing error: {str(e)}"
            }

        if "classification" not in result or "confidence" not in result or "reasoning" not in result:
            logger.warning(f"Missing required fields in Gemini response. Response: {result}")
            return {
                "classification": "General Information Enquiries|Unknown",
                "Question_Language": "en", 
                "Question_Script": "Latin",
                "confidence": "LOW",
                "reasoning": "Missing required fields in classification response."
            }
        
        result.setdefault("Question_Language", "en")
        result.setdefault("Question_Script", "Latin")

        classification = result.get("classification", "General Information Enquiries|Unknown")
        if "|" not in classification:
            # If the main category is returned without sub-category
            main_category = classification.strip()
            # Map defaults for some categories if needed
            default_sub = "Unknown"
            if main_category == "General":
                default_sub = "Greeting"  # For greetings, adjust as needed
            result["classification"] = f"{main_category}|{default_sub}"

        logger.info(f"Message classified as: {result.get('classification')} | Lang: {result.get('Question_Language')}:{result.get('Question_Script')} | Conf: {result.get('confidence')}")
        return result

    except Exception as e:
        logger.error(f"Gemini classification error: {e}")
        return {
            "classification": "General Information Enquiries|Unknown",
            "Question_Language": "en", 
            "Question_Script": "Latin",
            "confidence": "LOW",
            "reasoning": f"API error: {str(e)}"
        }

# ----------------------------
# Lifespan handler
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase, gemini_model
    logger.info("Starting FastAPI app on port {}", os.getenv('PORT', 10000))
    
    try:
        supabase = get_supabase_client()
        logger.info("Supabase Configuration: ✓ Set")
    except Exception as e:
        logger.error("Supabase connection failed: {}", e)
        supabase = None

    try:
        gemini_model = get_gemini_model()
        if gemini_model:
            logger.info("Gemini AI Configuration: ✓ Set")
        else:
            logger.warning("Gemini AI Configuration: ✗ Not available")
    except Exception as e:
        logger.error("Gemini AI initialization failed: {}", e)
        gemini_model = None

    yield
    logger.info("Application shutdown complete")

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(
    title="WhatsApp Message Processor with AI Classification",
    description="WhatsApp message processing service with AI classification and Supabase logging",
    version="1.0.6",
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
        problematic_keys = {k: type(v) for k, v in log_data.items() if not isinstance(v, (str, int, float, bool, list, dict, type(None)))}
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
    start_time = datetime.now(timezone.utc)
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
        "ai_confidence": None,
        "ai_reasoning": None,
    }

    classification_result = {
        "classification": "General Information Enquiries|Unknown",
        "confidence": "LOW",
        "reasoning": "Processing error or classification not performed"
    }
    final_message_to_classify = request.WA_Msg_Text
    response_data = None

    try:
        # --- Step 1: Check if message is partial ---
        completeness_check = await classify_message_completeness(request.WA_Msg_Text, gemini_model)

        # --- Step 2: Fetch history and consolidate if partial ---
        if completeness_check.completeness == "partial" and request.MobileNo and gemini_model:
            previous_messages = await fetch_previous_messages(request.MobileNo, request.WA_Message_Id, limit=10)
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


        # --- Step 3: Handle Images ---
        if request.WA_Msg_Type and request.WA_Msg_Type.lower() == "image" and request.WA_Url:
            logger.info(f"Received an image message from {phone_number}. Assigning placeholder classification.")
            # If you want to process image content, integrate analyze_image_with_gemini here.
            # For now, we assign a placeholder and skip text classification.
            classification_result = {
                "classification": "General Information Enquiries|Image Received",
                "confidence": "LOW",
                "reasoning": "Image message received, text classification skipped."
            }
            log_data["transcription"] = "Image message received (no text classification yet)"

        # --- Step 4: Classify the message ---
        elif final_message_to_classify and gemini_model:
            logger.info(f"Classifying message: {final_message_to_classify[:100]}...")
            classification_result = await classify_message_with_gemini(final_message_to_classify, gemini_model)

            # Ensure combined string format
            combined_classification = classification_result.get("classification", "")
            if not combined_classification or "|" not in combined_classification:
                combined_classification = "General Information Enquiries|Unknown"
            classification_result["classification"] = combined_classification

            if combined_classification.endswith("Unknown"):
                response_data = {
                    "phone_number": phone_number,
                    "ai_response": "Sorry, I didn’t understand your query properly. Can you rephrase the question or add more information?",
                    "ai_reason": "Classification failed even after consolidating previous messages",
                    "ai_classification": "General Information Enquiries|Unknown",
                }

                # Short-circuit return here so it doesn’t go to handlers
                end_time = datetime.now(timezone.utc)
                duration_ms = int((end_time - start_time).total_seconds() * 1000)
                log_data.update({
                    "status": "success",
                    "processing_end_time": end_time,
                    "processing_duration_ms": duration_ms,
                    "response_phone_number": response_data["phone_number"],
                    "response_ai_response": response_data["ai_response"],
                    "response_ai_reason": response_data["ai_reason"],
                    "raw_response": response_data,
                    "ai_classification": response_data["ai_classification"],
                })
                asyncio.create_task(log_to_supabase(log_data))
                return MessageResponse(
                    phone_number=response_data["phone_number"],
                    ai_response=response_data["ai_response"],
                    ai_reason=response_data["ai_reason"],
                    WA_Auto_Id=request.WA_Auto_Id,
                    WA_Message_Id=request.WA_Message_Id,
                )
        else:
            combined_classification = "General Information Enquiries|Unknown"

        log_data["ai_classification"] = combined_classification
        main_classification, sub_classification = combined_classification.split("|", 1)

        # --- Step 4: Route to Greeting or FAQ ---
        if main_classification == "General" and sub_classification == "Greeting":
            response_data = await handle_greeting(
                message_text=final_message_to_classify,
                user_name=request.Wa_Name or request.Donor_Name or "Sevak",
                classification_result={"classification": combined_classification},
                phone_number=phone_number
            )


        else:
            answer = await handle_faq(final_message_to_classify)
            response_data = {
                "phone_number": phone_number,
                "ai_response": answer,
                "ai_reason": f"FAQ answer for classification {combined_classification}",
                "ai_classification": combined_classification,
            }


            if not response_data or not response_data.get("ai_response"):
                response_data = {
                    "phone_number": phone_number,
                    "ai_response": "Sorry, I don’t have that answer.",
                    "ai_reason": f"No FAQ match found for: {combined_classification}",
                    "ai_classification": combined_classification,
                }

        # Attach WA IDs
        if request.WA_Auto_Id is not None:
            response_data["WA_Auto_Id"] = request.WA_Auto_Id
        if request.WA_Message_Id is not None:
            response_data["WA_Message_Id"] = request.WA_Message_Id

        # --- Finalize logs ---
        end_time = datetime.now(timezone.utc)
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
            "raw_response": response_data,
            "ai_classification": combined_classification,
        })

        asyncio.create_task(log_to_supabase(log_data))
        logger.info(f"Request {request_id} processed in {duration_ms}ms.")

        return MessageResponse(
            phone_number=response_data.get("phone_number", phone_number),
            ai_response=response_data.get("ai_response", "Error generating response."),
            ai_reason=response_data.get("ai_reason", "N/A"),
            WA_Auto_Id=response_data.get("WA_Auto_Id"),
            WA_Message_Id=response_data.get("WA_Message_Id"),
        )

    except Exception as e:
        end_time = datetime.now(timezone.utc)
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        log_data.update({
            "status": "error",
            "processing_end_time": end_time,
            "processing_duration_ms": duration_ms,
            "error_type": "internal_error",
            "error_message": str(e),
            "raw_response": {"error": str(e)}
        })
        asyncio.create_task(log_to_supabase(log_data))
        logger.error(f"Request {request_id} failed after {duration_ms}ms: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

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
    if not gemini_model:
        raise HTTPException(status_code=503, detail="Gemini AI not available")
    if is_partial_check_needed:
        completeness_result = await classify_message_completeness(message, gemini_model)
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

    result = await classify_message_with_gemini(message, gemini_model)
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
        status="healthy" if supabase and gemini_model else "degraded",
        timestamp=datetime.now().isoformat(),
        service="whatsapp-message-processor",
        version="1.0.1"
    )

@app.get("/metrics")
async def metrics():
    return {
        "service": "whatsapp-message-processor",
        "supabase_enabled": supabase is not None,
        "gemini_ai_enabled": gemini_model is not None,
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


async def classify_message_completeness(message_text: str, gemini_model) -> MessageCompletenessResponse:
    """
    Use Gemini to determine if a message is a complete thought or a partial one.
    This is useful for multi-line inputs like addresses.
    """
    if not gemini_model:
        logger.warning("Gemini client not available. Assuming message is 'full'.")
        return MessageCompletenessResponse(completeness="full", reasoning="Gemini client not available")

    if not message_text or message_text.strip() == "":
        return MessageCompletenessResponse(completeness="full", reasoning="Empty message") # Treat empty as full (nothing to process)

    prompt = f"""
You are a message analysis system. Your task is to determine if the provided text represents a complete thought or a partial message that is likely part of a larger input (like a multi-line address or a sentence being typed).

Consider the following:
- A "full" message is a complete sentence, a standalone question, a statement, or a greeting that doesn't obviously require more context to be understood.
- A "partial" message is a fragment that seems incomplete, often like the beginning or middle of a sentence, or a list item that's not the last one.

Analyze the following message: "{message_text}"

Respond ONLY with a JSON object containing two keys:
"completeness": "full" or "partial"
"reasoning": A brief explanation for your decision.

Example 1:
Message: "Hello there!"
JSON: {{"completeness": "full", "reasoning": "It's a standard greeting."}}

Example 2:
Message: "123 Main Street"
JSON: {{"completeness": "partial", "reasoning": "This looks like the first line of an address, likely followed by city and zip code."}}

Example 3:
Message: "I want to donate"
JSON: {{"completeness": "partial", "reasoning": "This sentence is incomplete and likely a precursor to specifying the donation amount or purpose."}}

Example 4:
Message: "Thank you for your help."
JSON: {{"completeness": "full", "reasoning": "This is a complete and polite closing statement."}}

Now, provide the JSON for the message above.
"""

    try:
        response = await gemini_model.generate_content_async(prompt)
        result_text = response.text.strip()

        # Clean up markdown formatting
        if result_text.startswith("```json"):
            result_text = result_text.replace("```json", "").replace("```", "").strip()

        result = json.loads(result_text)
        logger.info(f"Message completeness check: {result.get('completeness')} - {result.get('reasoning')}")
        return MessageCompletenessResponse(**result)

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error from Gemini completeness check: {e}")
        return MessageCompletenessResponse(completeness="full", reasoning=f"JSON parsing error: {str(e)}")
    except Exception as e:
        logger.error(f"Gemini API error during completeness check: {e}")
        return MessageCompletenessResponse(completeness="full", reasoning=f"Gemini API error: {str(e)}")
    

# Placeholder for fetching previous messages
# In a real app, this would query a database (e.g., Supabase)
# ----------------------------
async def fetch_previous_messages(mobile_no: str, current_received_at: Optional[datetime], limit: int = 10) -> List[str]:
    """
    Fetches the last 'limit' messages for a given mobile number that were received BEFORE the current message.
    """
    if not supabase:
        logger.warning("Supabase not initialized. Cannot fetch previous messages.")
        return []
    
    if not current_received_at:
        logger.warning("Cannot fetch previous messages without current_received_at timestamp.")
        return []

    logger.info(f"Attempting to fetch last {limit} messages for {mobile_no} received before {current_received_at.isoformat()}")
    try:
        query = (
            supabase.table("message_logs")
            .select("wa_msg_text")
            .eq("mobile_no", mobile_no)
            .lt("wa_received_at", current_received_at.isoformat())
            .order("wa_received_at", ascending=False)
            .limit(limit)
        )
        
        response = query.execute()

        if response.data:
            messages = [msg["wa_msg_text"] for msg in response.data if msg["wa_msg_text"]]
            return messages[::-1] # Reverse to get chronological order
        else:
            logger.info(f"No previous messages found for {mobile_no} before {current_received_at.isoformat()}")
            return []
    except Exception as e:
        logger.error(f"Error fetching previous messages from Supabase for {mobile_no}: {e}")
        return []
    

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
