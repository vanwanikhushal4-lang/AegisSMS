# -*- coding: utf-8 -*-
"""
Synthetic SMS dataset generator for phishing/smishing detection.
Produces 4 separate CSV files (English, Hinglish, Hindi, Marathi), each with
columns LABEL,TEXT,URL,EMAIL,PHONE matching the schema of Dataset_5971.csv.

Labels: ham, spam, Smishing (even ~33/33/33 split per language file).
Content is built from hand-written templates + randomized slot-filling
(names, banks, amounts, OTPs, dates, URLs, etc.) so that text is highly
varied even at large scale.
"""

import csv
import os
import random
import string

random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset_5971")
TOTAL_PER_LANG = 135000  # ~5.4 lakh total across 4 language files
LABELS = ["ham", "spam", "Smishing"]

# ---------------------------------------------------------------------------
# Shared entity pools (brand/entity names stay in Latin script even inside
# Hindi/Marathi sentences -- this matches how real Indian SMS mix scripts).
# ---------------------------------------------------------------------------
BANKS = ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "Punjab National Bank",
         "Bank of Baroda", "Kotak Mahindra Bank", "Canara Bank",
         "Union Bank of India", "IDBI Bank", "Yes Bank", "IndusInd Bank",
         "Bank of India", "Central Bank of India"]
COMPANIES = ["Amazon", "Flipkart", "Myntra", "Meesho", "Ajio", "Snapdeal",
             "BigBasket", "Nykaa", "Zomato", "Swiggy"]
COURIERS = ["Blue Dart", "Delhivery", "DTDC", "India Post", "Ekart", "FedEx",
            "Xpressbees", "Shadowfax"]
WALLETS = ["Paytm", "PhonePe", "Google Pay", "Amazon Pay", "Mobikwik"]
TELECOM = ["Airtel", "Jio", "Vi", "BSNL"]
GOVT = ["Income Tax Department", "EPFO", "UIDAI", "Election Commission of India",
        "Passport Seva"]

NAMES = {
    "english": ["Raj", "Priya", "Amit", "Sneha", "Rahul", "Anita", "Vikram",
                "Pooja", "Suresh", "Kavita", "Arjun", "Neha", "Karan", "Divya",
                "Rohan", "Meera", "Sanjay", "Anjali", "Vivek", "Shreya",
                "Manoj", "Ritu", "Ajay", "Swati", "Nikhil", "Preeti", "Deepak",
                "Kiran", "Rakesh", "Sunita"],
    "hinglish": ["Raj", "Priya", "Amit", "Sneha", "Rahul", "Anita", "Vikram",
                 "Pooja", "Suresh", "Kavita", "Arjun", "Neha", "Karan", "Divya",
                 "Rohan", "Meera", "Sanjay", "Anjali", "Vivek", "Shreya",
                 "Manoj", "Ritu", "Ajay", "Swati", "Nikhil", "Preeti", "Deepak",
                 "Kiran", "Rakesh", "Sunita"],
    "hindi": ["राज", "प्रिया", "अमित", "स्नेहा", "राहुल", "अनीता", "विक्रम",
              "पूजा", "सुरेश", "कविता", "अर्जुन", "नेहा", "करण", "दिव्या",
              "रोहन", "मीरा", "संजय", "अंजलि", "विवेक", "श्रेया", "मनोज",
              "रितु", "अजय", "स्वाति", "निखिल", "प्रीति", "दीपक", "किरण",
              "राकेश", "सुनीता"],
    "marathi": ["राज", "प्रिया", "अमोल", "स्नेहल", "राहुल", "अनिता", "विक्रम",
                "पूजा", "सुरेश", "कविता", "अर्जुन", "नेहा", "करण", "दिव्या",
                "रोहन", "मीरा", "संजय", "अंजली", "विवेक", "श्रेया", "मनोज",
                "रितू", "अजय", "स्वाती", "निखिल", "प्रीती", "दीपक", "किरण",
                "राकेश", "सुनीता"],
}

CITIES = {
    "english": ["Mumbai", "Delhi", "Pune", "Bengaluru", "Chennai", "Hyderabad",
                "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Nagpur", "Nashik",
                "Indore", "Surat"],
    "hinglish": ["Mumbai", "Delhi", "Pune", "Bengaluru", "Chennai", "Hyderabad",
                 "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Nagpur", "Nashik",
                 "Indore", "Surat"],
    "hindi": ["मुंबई", "दिल्ली", "पुणे", "बेंगलुरु", "चेन्नई", "हैदराबाद",
              "कोलकाता", "अहमदाबाद", "जयपुर", "लखनऊ", "नागपुर", "नासिक",
              "इंदौर", "सूरत"],
    "marathi": ["मुंबई", "पुणे", "नागपूर", "नाशिक", "औरंगाबाद", "कोल्हापूर",
                "सोलापूर", "ठाणे", "सातारा", "सांगली"],
}

# ---------------------------------------------------------------------------
# Random slot generators
# ---------------------------------------------------------------------------
def rand_otp():
    length = random.choice([4, 6])
    return ''.join(random.choices(string.digits, k=length))


def rand_amount():
    value = random.choice([random.randint(50, 999), random.randint(1000, 9999),
                           random.randint(10000, 99999), random.randint(100000, 999999)])
    return f"{value:,}"


def rand_phone():
    first = random.choice("6789")
    rest = ''.join(random.choices(string.digits, k=9))
    return f"+91{first}{rest}"


def rand_last4():
    return ''.join(random.choices(string.digits, k=4))


def rand_orderid():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))


def rand_tid():
    return ''.join(random.choices(string.digits, k=10))


def rand_date():
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.choice([2025, 2026])
    return f"{day:02d}-{month:02d}-{year}"


def rand_time():
    hour = random.randint(1, 12)
    minute = random.randint(0, 59)
    ap = random.choice(["AM", "PM"])
    return f"{hour}:{minute:02d} {ap}"


def rand_month():
    return random.choice(["January", "February", "March", "April", "May", "June",
                          "July", "August", "September", "October", "November", "December"])


def rand_url():
    domains = ["bit.ly", "tinyurl.com", "rebrand.ly", "cutt.ly", "shorturl.at"]
    suspicious_domains = ["kyc-verify-secure.info", "bank-alert-update.com",
                          "account-verify24.xyz", "offer4u-claim.co",
                          "secure-login-update.net", "refund-claim.online",
                          "paytm-kyc-update.co", "parcel-redelivery.xyz"]
    if random.random() < 0.5:
        suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=7))
        return f"http://{random.choice(domains)}/{suffix}"
    else:
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
        return f"http://{random.choice(suspicious_domains)}/{suffix}"


def rand_email():
    user = ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 8)))
    domain = random.choice(["gmail.com", "yahoo.com", "outlook.com",
                            "support-team.com", "customercare.co.in"])
    return f"{user}@{domain}"


def fill(template, names, cities):
    return template.format(
        bank=random.choice(BANKS),
        company=random.choice(COMPANIES),
        courier=random.choice(COURIERS),
        wallet=random.choice(WALLETS),
        telecom=random.choice(TELECOM),
        govt=random.choice(GOVT),
        name=random.choice(names),
        city=random.choice(cities),
        amount=rand_amount(),
        amount2=rand_amount(),
        otp=rand_otp(),
        phone=rand_phone(),
        last4=rand_last4(),
        orderid=rand_orderid(),
        tid=rand_tid(),
        date=rand_date(),
        time=rand_time(),
        month=rand_month(),
        url=rand_url(),
    )


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
TEMPLATES = {
    "english": {
        "ham": [
            "Hi {name}, your OTP for login is {otp}. Valid for 10 minutes. Do not share this with anyone.",
            "Dear Customer, Rs.{amount} debited from your {bank} account ending {last4} on {date}. Avl Bal: Rs.{amount2}.",
            "Your {courier} package (Tracking ID {tid}) was delivered today at {time}. Thank you for choosing us.",
            "Reminder: Your appointment with Dr. {name} is on {date} at {time}. Please arrive 10 mins early.",
            "Hey {name}, are we still meeting for lunch at {time} today?",
            "Your {telecom} prepaid recharge of Rs.{amount} was successful. Validity till {date}.",
            "Thank you for shopping with {company}. Your order #{orderid} has been shipped and will arrive by {date}.",
            "Hi, just checking in - how did your exam go last week?",
            "Meeting rescheduled to {time} on {date}. Let me know if that works for you.",
            "Your electricity bill of Rs.{amount} for {month} has been paid successfully. Thank you.",
            "Hi {name}, can you send me the notes from today's class?",
            "Your {wallet} wallet has been credited with Rs.{amount} cashback. Enjoy your rewards!",
            "Dear Customer, your {bank} credit card payment of Rs.{amount} is due on {date}.",
            "Happy Birthday {name}! Hope you have a wonderful year ahead.",
            "Your booking with {company} for {date} is confirmed. PNR: {orderid}.",
            "Mom, I'll be home by {time}, don't wait for dinner.",
            "Your {telecom} bill payment of Rs.{amount} was successful. Thank you for using our services.",
            "Can we catch up this weekend? It's been a while since we talked.",
            "Your {courier} parcel is out for delivery and will reach you by {time} today.",
            "Congrats on the new job, {name}! Let's celebrate soon.",
        ],
        "spam": [
            "Congratulations! You've been selected for a special {company} discount. Get 50% off your next purchase. Shop now!",
            "Limited period offer! Get instant personal loan up to Rs.{amount} with minimal documentation. Apply today.",
            "Win exciting prizes every week! Reply WIN to participate. T&C apply.",
            "Get 2 free movie tickets with your {telecom} recharge of Rs.{amount} this week only.",
            "Exclusive insurance plan for just Rs.{amount}/month. Cover your family today. Call {phone} to know more.",
            "Earn Rs.{amount} daily working from home. No investment needed. Reply YES to start.",
            "Flat 70% off on {company} - today only! Use code SAVE70 at checkout.",
            "Hot singles in {city} want to chat with you! Reply NOW to connect.",
            "Get a free credit card with zero annual fee. Apply now and get Rs.{amount} welcome bonus.",
            "Your {telecom} number is eligible for a free upgrade to 5G. Reply UPGRADE to activate.",
            "Best forex trading tips! Turn Rs.{amount} into Rs.{amount2} in a month. Call {phone} now.",
            "Buy 1 Get 1 Free on all {company} products this weekend only. Hurry, offer ends soon!",
            "Get rid of your debt today! Consolidate loans starting at low interest. Call {phone}.",
            "Lose 10kg in 30 days with our special diet plan. Reply INFO to learn more.",
            "Your astrology reading says big changes are coming! Call {phone} for a free consultation.",
            "Congratulations! You qualify for a pre-approved loan of Rs.{amount}. Reply APPLY now.",
            "New year sale at {company}! Flat 60% off storewide. Shop now before stock runs out.",
            "Book your dream vacation to {city} at unbelievable prices. Call {phone} for packages.",
            "Get instant approval for your credit card, no income proof required. Reply YES.",
            "Free recharge of Rs.{amount} on referring 3 friends to our app. Reply REFER now.",
        ],
        "Smishing": [
            "Dear Customer, your {bank} account will be blocked today. Update your KYC immediately: {url}",
            "Your {courier} parcel is on hold due to unpaid customs fee of Rs.{amount}. Pay now: {url}",
            "Congratulations! Your mobile number has won Rs.{amount} in the {company} lucky draw. Claim now: {url}",
            "URGENT: Your {bank} debit card has been temporarily suspended. Verify your details here: {url}",
            "Your Aadhaar card will be deactivated. Link with PAN immediately at {url} to avoid penalty.",
            "Income Tax Refund of Rs.{amount} is pending. Click {url} to claim before {date}.",
            "Your electricity connection will be disconnected tonight due to unpaid bill. Pay immediately: {url}",
            "Dear user, your {wallet} KYC has expired. Update now or your wallet will be blocked: {url}",
            "You have been selected for a job at {company} with salary Rs.{amount}/month. Register now: {url}",
            "Your parcel could not be delivered. Reschedule and pay Rs.{amount} shipping fee at {url}",
            "{bank} Alert: Unusual login detected on your account. Secure it now: {url}",
            "Your {wallet} account is temporarily locked. Verify your identity to unlock: {url}",
            "Dear applicant, your {govt} document is ready. Download and pay processing fee at {url}",
            "Your SIM card will be blocked in 24 hours. Update your {telecom} KYC now: {url}",
            "Final notice: Your {bank} account will be suspended due to pending verification. Click: {url}",
            "Your credit score has dropped! Check and fix it instantly at {url}",
            "You've received a payment of Rs.{amount} via {wallet}. Click here to accept: {url}",
            "Dear customer, complete your annual bank locker KYC or it will be sealed: {url}",
            "Your {company} account shows suspicious activity. Verify now to avoid permanent suspension: {url}",
            "Congratulations {name}! You've won an iPhone from {company}. Claim your prize: {url}",
        ],
    },
    "hinglish": {
        "ham": [
            "Hi {name}, aapka login OTP {otp} hai. Yeh 10 minute ke liye valid hai. Kisi ke saath share mat karna.",
            "Priya Grahak, aapke {bank} account se {date} ko Rs.{amount} debit hua hai. Available balance: Rs.{amount2}.",
            "Aapka {courier} package (Tracking ID {tid}) aaj {time} baje deliver ho gaya hai. Dhanyavad.",
            "Reminder: Dr. {name} ke saath aapki appointment {date} ko {time} baje hai. 10 minute pehle pahunchein.",
            "Hey {name}, kya hum aaj {time} baje lunch ke liye mil rahe hain?",
            "Aapka {telecom} prepaid recharge Rs.{amount} ka successful hua hai. Validity {date} tak.",
            "{company} se shopping karne ke liye dhanyavad. Aapka order #{orderid} ship ho chuka hai.",
            "Hi, bas check kar raha tha - pichle hafte exam kaisa gaya?",
            "Meeting {date} ko {time} baje reschedule ho gayi hai. Batana agar yeh time thik hai.",
            "Aapka {month} ka electricity bill Rs.{amount} ka successfully pay ho gaya hai. Dhanyavad.",
            "Hi {name}, aaj ki class ke notes bhej sakte ho kya?",
            "Aapke {wallet} wallet mein Rs.{amount} cashback credit hua hai. Enjoy karein!",
            "Priya Grahak, aapke {bank} credit card ka Rs.{amount} payment {date} ko due hai.",
            "Happy Birthday {name}! Umeed hai tumhara saal bahut accha rahega.",
            "{company} ke saath aapki booking {date} ke liye confirm ho gayi hai. PNR: {orderid}.",
            "Mom, main {time} tak ghar aa jaunga, dinner ke liye wait mat karna.",
            "Aapka {telecom} bill payment Rs.{amount} ka successful hua hai. Dhanyavad.",
            "Is weekend milte hain kya? Bahut time ho gaya baat kiye.",
            "Aapka {courier} parcel out for delivery hai aur aaj {time} tak pahunch jayega.",
            "Naye job ke liye badhai ho {name}! Jald hi celebrate karte hain.",
        ],
        "spam": [
            "Badhai ho! Aapko {company} ka special discount mila hai. Agli purchase par 50% off. Abhi shop karein!",
            "Limited period offer! Rs.{amount} tak ka instant personal loan, minimal documents ke saath. Aaj hi apply karein.",
            "Har hafte exciting prizes jeetein! Participate karne ke liye WIN reply karein. T&C apply.",
            "{telecom} ke Rs.{amount} recharge ke saath 2 free movie tickets paayein, sirf is hafte.",
            "Sirf Rs.{amount}/month mein exclusive insurance plan. Apni family ko protect karein. Call karein {phone}.",
            "Ghar baithe roz Rs.{amount} kamayein. Koi investment nahi. Shuru karne ke liye YES reply karein.",
            "{company} par flat 70% off - sirf aaj! Checkout par SAVE70 code use karein.",
            "{city} mein singles aapse chat karna chahte hain! Abhi NOW reply karke connect karein.",
            "Zero annual fee wala free credit card paayein. Abhi apply karein aur Rs.{amount} welcome bonus paayein.",
            "Aapka {telecom} number free 5G upgrade ke liye eligible hai. Activate karne ke liye UPGRADE reply karein.",
            "Best forex trading tips! Rs.{amount} ko ek mahine mein Rs.{amount2} banayein. Abhi call karein {phone}.",
            "Is weekend sirf {company} ke saare products par Buy 1 Get 1 Free. Jaldi karein, offer khatam hone wala hai!",
            "Apna debt aaj hi khatam karein! Kam interest rate par loans consolidate karein. Call karein {phone}.",
            "30 dino mein 10kg wajan kam karein humare special diet plan se. Jaankari ke liye INFO reply karein.",
            "Aapki astrology reading kehti hai badi changes aane wali hain! Free consultation ke liye call karein {phone}.",
            "Badhai ho! Aap Rs.{amount} ke pre-approved loan ke liye qualify karte hain. Abhi APPLY reply karein.",
            "{company} par New Year sale! Storewide flat 60% off. Stock khatam hone se pehle shop karein.",
            "{city} ke liye apni dream vacation book karein bemisaal prices par. Packages ke liye call karein {phone}.",
            "Apne credit card ke liye instant approval paayein, koi income proof nahi chahiye. YES reply karein.",
            "3 dosto ko refer karke Rs.{amount} ka free recharge paayein. Abhi REFER reply karein.",
        ],
        "Smishing": [
            "Priya Grahak, aapka {bank} account aaj block ho jayega. Turant KYC update karein: {url}",
            "Aapka {courier} parcel customs fee Rs.{amount} bakaya hone ki wajah se hold par hai. Abhi pay karein: {url}",
            "Badhai ho! Aapka mobile number {company} lucky draw mein Rs.{amount} jeet chuka hai. Abhi claim karein: {url}",
            "URGENT: Aapka {bank} debit card temporarily suspend ho gaya hai. Apni details yahan verify karein: {url}",
            "Aapka Aadhaar card deactivate ho jayega. Penalty se bachne ke liye {url} par turant PAN se link karein.",
            "Rs.{amount} ka Income Tax Refund pending hai. {date} se pehle claim karne ke liye {url} par click karein.",
            "Aapka electricity connection aaj raat bakaya bill ki wajah se disconnect ho jayega. Turant pay karein: {url}",
            "Priya user, aapki {wallet} KYC expire ho gayi hai. Abhi update karein warna wallet block ho jayega: {url}",
            "Aapko {company} mein Rs.{amount}/month salary wali job ke liye select kiya gaya hai. Abhi register karein: {url}",
            "Aapka parcel deliver nahi ho paya. Rs.{amount} shipping fee pay karke reschedule karein: {url}",
            "{bank} Alert: Aapke account mein unusual login detect hua hai. Isse abhi secure karein: {url}",
            "Aapka {wallet} account temporarily lock ho gaya hai. Unlock karne ke liye apni identity verify karein: {url}",
            "Priya applicant, aapka {govt} document ready hai. Download karein aur processing fee pay karein: {url}",
            "Aapka SIM card 24 ghante mein block ho jayega. Abhi apni {telecom} KYC update karein: {url}",
            "Final notice: Pending verification ki wajah se aapka {bank} account suspend ho jayega. Click karein: {url}",
            "Aapka credit score gir gaya hai! Ise abhi {url} par check aur fix karein.",
            "Aapko {wallet} se Rs.{amount} ka payment mila hai. Accept karne ke liye yahan click karein: {url}",
            "Priya customer, apni annual bank locker KYC complete karein warna woh seal ho jayega: {url}",
            "Aapke {company} account mein suspicious activity dikh rahi hai. Permanent suspension se bachne ke liye abhi verify karein: {url}",
            "Badhai ho {name}! Aapne {company} se iPhone jeeta hai. Apna prize claim karein: {url}",
        ],
    },
    "hindi": {
        "ham": [
            "प्रिय {name}, आपके लॉगिन के लिए OTP {otp} है। यह 10 मिनट के लिए मान्य है। कृपया इसे किसी के साथ साझा न करें।",
            "प्रिय ग्राहक, आपके {bank} खाते से {date} को Rs.{amount} डेबिट हुए हैं। उपलब्ध शेष: Rs.{amount2}।",
            "आपका {courier} पार्सल (ट्रैकिंग आईडी {tid}) आज {time} बजे डिलीवर हो गया है। धन्यवाद।",
            "अनुस्मारक: डॉ. {name} के साथ आपकी अपॉइंटमेंट {date} को {time} बजे है। कृपया 10 मिनट पहले पहुंचें।",
            "अरे {name}, क्या हम आज {time} बजे लंच के लिए मिल रहे हैं?",
            "आपका {telecom} प्रीपेड रिचार्ज Rs.{amount} सफल रहा। वैधता {date} तक।",
            "{company} से खरीदारी करने के लिए धन्यवाद। आपका ऑर्डर #{orderid} शिप हो चुका है।",
            "हाय, बस पूछ रहा था - पिछले हफ्ते परीक्षा कैसी रही?",
            "मीटिंग {date} को {time} बजे के लिए पुनर्निर्धारित हो गई है। बताइए क्या यह समय ठीक है।",
            "आपका {month} का बिजली बिल Rs.{amount} सफलतापूर्वक भुगतान हो गया है। धन्यवाद।",
            "हाय {name}, क्या तुम आज की क्लास के नोट्स भेज सकते हो?",
            "आपके {wallet} वॉलेट में Rs.{amount} कैशबैक जमा हुआ है। आनंद लें!",
            "प्रिय ग्राहक, आपके {bank} क्रेडिट कार्ड का Rs.{amount} भुगतान {date} को देय है।",
            "जन्मदिन मुबारक हो {name}! आपका आने वाला साल शानदार रहे।",
            "{company} के साथ आपकी बुकिंग {date} के लिए पुष्टि हो गई है। पीएनआर: {orderid}।",
            "मम्मी, मैं {time} तक घर आ जाऊंगा, डिनर के लिए इंतज़ार मत करना।",
            "आपका {telecom} बिल भुगतान Rs.{amount} सफल रहा। हमारी सेवाओं का उपयोग करने के लिए धन्यवाद।",
            "इस वीकेंड मिलते हैं क्या? बात किए हुए काफी समय हो गया।",
            "आपका {courier} पार्सल डिलीवरी के लिए निकल चुका है और आज {time} तक पहुंच जाएगा।",
            "नई नौकरी की बधाई हो {name}! जल्द ही जश्न मनाते हैं।",
        ],
        "spam": [
            "बधाई हो! आपको {company} का विशेष डिस्काउंट मिला है। अगली खरीद पर 50% छूट। अभी खरीदारी करें!",
            "सीमित समय का ऑफर! न्यूनतम दस्तावेज़ों के साथ Rs.{amount} तक का इंस्टेंट पर्सनल लोन पाएं। आज ही आवेदन करें।",
            "हर हफ्ते रोमांचक इनाम जीतें! भाग लेने के लिए WIN रिप्लाई करें। नियम व शर्तें लागू।",
            "{telecom} के Rs.{amount} रिचार्ज के साथ पाएं 2 मुफ्त मूवी टिकट, सिर्फ इस हफ्ते।",
            "सिर्फ Rs.{amount}/माह में एक्सक्लूसिव इंश्योरेंस प्लान। अपने परिवार को सुरक्षित करें। कॉल करें {phone}।",
            "घर बैठे रोज़ाना Rs.{amount} कमाएं। कोई निवेश नहीं। शुरू करने के लिए YES रिप्लाई करें।",
            "{company} पर फ्लैट 70% छूट - सिर्फ आज! चेकआउट पर SAVE70 कोड इस्तेमाल करें।",
            "{city} में सिंगल्स आपसे चैट करना चाहते हैं! अभी NOW रिप्लाई करके जुड़ें।",
            "ज़ीरो एनुअल फीस वाला फ्री क्रेडिट कार्ड पाएं। अभी आवेदन करें और Rs.{amount} वेलकम बोनस पाएं।",
            "आपका {telecom} नंबर फ्री 5G अपग्रेड के लिए योग्य है। एक्टिवेट करने के लिए UPGRADE रिप्लाई करें।",
            "बेस्ट फॉरेक्स ट्रेडिंग टिप्स! Rs.{amount} को एक महीने में Rs.{amount2} बनाएं। अभी कॉल करें {phone}।",
            "इस वीकेंड सिर्फ {company} के सभी प्रोडक्ट्स पर Buy 1 Get 1 Free। जल्दी करें, ऑफर खत्म होने वाला है!",
            "अपना कर्ज़ आज ही खत्म करें! कम ब्याज दर पर लोन कंसोलिडेट करें। कॉल करें {phone}।",
            "हमारे स्पेशल डाइट प्लान से 30 दिनों में 10 किलो वज़न घटाएं। जानकारी के लिए INFO रिप्लाई करें।",
            "आपकी ज्योतिष रीडिंग कहती है बड़े बदलाव आने वाले हैं! मुफ्त परामर्श के लिए कॉल करें {phone}।",
            "बधाई हो! आप Rs.{amount} के प्री-अप्रूव्ड लोन के लिए योग्य हैं। अभी APPLY रिप्लाई करें।",
            "{company} पर न्यू ईयर सेल! स्टोरवाइड फ्लैट 60% छूट। स्टॉक खत्म होने से पहले खरीदारी करें।",
            "{city} के लिए अपनी ड्रीम वेकेशन बेमिसाल कीमतों पर बुक करें। पैकेज के लिए कॉल करें {phone}।",
            "अपने क्रेडिट कार्ड के लिए इंस्टेंट अप्रूवल पाएं, कोई इनकम प्रूफ नहीं चाहिए। YES रिप्लाई करें।",
            "3 दोस्तों को रेफर करके Rs.{amount} का फ्री रिचार्ज पाएं। अभी REFER रिप्लाई करें।",
        ],
        "Smishing": [
            "प्रिय ग्राहक, आपका {bank} खाता आज ब्लॉक हो जाएगा। तुरंत केवाईसी अपडेट करें: {url}",
            "आपका {courier} पार्सल Rs.{amount} बकाया कस्टम शुल्क के कारण होल्ड पर है। अभी भुगतान करें: {url}",
            "बधाई हो! आपका मोबाइल नंबर {company} लकी ड्रॉ में Rs.{amount} जीत चुका है। अभी दावा करें: {url}",
            "अत्यावश्यक: आपका {bank} डेबिट कार्ड अस्थायी रूप से निलंबित कर दिया गया है। यहां अपनी जानकारी सत्यापित करें: {url}",
            "आपका आधार कार्ड निष्क्रिय हो जाएगा। जुर्माने से बचने के लिए तुरंत {url} पर पैन से लिंक करें।",
            "Rs.{amount} का इनकम टैक्स रिफंड लंबित है। {date} से पहले दावा करने के लिए {url} पर क्लिक करें।",
            "बकाया बिल के कारण आपका बिजली कनेक्शन आज रात काट दिया जाएगा। तुरंत भुगतान करें: {url}",
            "प्रिय उपयोगकर्ता, आपकी {wallet} केवाईसी समाप्त हो गई है। अभी अपडेट करें वरना वॉलेट ब्लॉक हो जाएगा: {url}",
            "आपको {company} में Rs.{amount}/माह वेतन वाली नौकरी के लिए चुना गया है। अभी रजिस्टर करें: {url}",
            "आपका पार्सल डिलीवर नहीं हो सका। Rs.{amount} शिपिंग शुल्क भुगतान करके पुनर्निर्धारित करें: {url}",
            "{bank} अलर्ट: आपके खाते में असामान्य लॉगिन का पता चला है। अभी इसे सुरक्षित करें: {url}",
            "आपका {wallet} खाता अस्थायी रूप से लॉक हो गया है। अनलॉक करने के लिए अपनी पहचान सत्यापित करें: {url}",
            "प्रिय आवेदक, आपका {govt} दस्तावेज़ तैयार है। डाउनलोड करें और प्रोसेसिंग शुल्क भुगतान करें: {url}",
            "आपका सिम कार्ड 24 घंटे में ब्लॉक हो जाएगा। अभी अपनी {telecom} केवाईसी अपडेट करें: {url}",
            "अंतिम सूचना: लंबित सत्यापन के कारण आपका {bank} खाता निलंबित कर दिया जाएगा। क्लिक करें: {url}",
            "आपका क्रेडिट स्कोर गिर गया है! इसे अभी {url} पर जांचें और ठीक करें।",
            "आपको {wallet} के माध्यम से Rs.{amount} का भुगतान प्राप्त हुआ है। स्वीकार करने के लिए यहां क्लिक करें: {url}",
            "प्रिय ग्राहक, अपनी वार्षिक बैंक लॉकर केवाईसी पूरी करें वरना यह सील कर दिया जाएगा: {url}",
            "आपके {company} खाते में संदिग्ध गतिविधि दिखाई दे रही है। स्थायी निलंबन से बचने के लिए अभी सत्यापित करें: {url}",
            "बधाई हो {name}! आपने {company} से iPhone जीता है। अपना इनाम प्राप्त करें: {url}",
        ],
    },
    "marathi": {
        "ham": [
            "प्रिय {name}, तुमच्या लॉगिनसाठी OTP {otp} आहे. हा 10 मिनिटांसाठी वैध आहे. कृपया कोणाशीही शेअर करू नका.",
            "प्रिय ग्राहक, तुमच्या {bank} खात्यातून {date} रोजी Rs.{amount} डेबिट झाले आहेत. उपलब्ध शिल्लक: Rs.{amount2}.",
            "तुमचे {courier} पार्सल (ट्रॅकिंग आयडी {tid}) आज {time} वाजता डिलिव्हर झाले आहे. धन्यवाद.",
            "स्मरणपत्र: डॉ. {name} यांच्यासोबत तुमची अपॉइंटमेंट {date} रोजी {time} वाजता आहे. कृपया 10 मिनिटे आधी पोहोचा.",
            "अरे {name}, आपण आज {time} वाजता जेवणासाठी भेटतोय का?",
            "तुमचा {telecom} प्रीपेड रिचार्ज Rs.{amount} यशस्वी झाला. वैधता {date} पर्यंत.",
            "{company} कडून खरेदी केल्याबद्दल धन्यवाद. तुमची ऑर्डर #{orderid} पाठवण्यात आली आहे.",
            "हाय, सहज विचारत होतो - मागच्या आठवड्यातली परीक्षा कशी झाली?",
            "मीटिंग {date} रोजी {time} वाजेपर्यंत पुढे ढकलली आहे. सांगा ही वेळ ठीक आहे का.",
            "तुमचे {month} महिन्याचे वीज बिल Rs.{amount} यशस्वीरित्या भरले गेले आहे. धन्यवाद.",
            "हाय {name}, आजच्या वर्गाच्या नोट्स पाठवशील का?",
            "तुमच्या {wallet} वॉलेटमध्ये Rs.{amount} कॅशबॅक जमा झाले आहे. आनंद घ्या!",
            "प्रिय ग्राहक, तुमच्या {bank} क्रेडिट कार्डचे Rs.{amount} पेमेंट {date} रोजी देय आहे.",
            "वाढदिवसाच्या शुभेच्छा {name}! तुझे येणारे वर्ष खूप छान जावो.",
            "{company} सोबतची तुमची बुकिंग {date} साठी कन्फर्म झाली आहे. पीएनआर: {orderid}.",
            "आई, मी {time} पर्यंत घरी येईन, जेवणासाठी थांबू नकोस.",
            "तुमचे {telecom} बिल पेमेंट Rs.{amount} यशस्वी झाले. आमच्या सेवा वापरल्याबद्दल धन्यवाद.",
            "या वीकेंडला भेटूया का? खूप दिवस झाले बोलून.",
            "तुमचे {courier} पार्सल डिलिव्हरीसाठी निघाले आहे आणि आज {time} पर्यंत पोहोचेल.",
            "नवीन नोकरीबद्दल अभिनंदन {name}! लवकरच सेलिब्रेट करूया.",
        ],
        "spam": [
            "अभिनंदन! तुम्हाला {company} चा खास डिस्काउंट मिळाला आहे. पुढच्या खरेदीवर 50% सूट. आत्ताच खरेदी करा!",
            "मर्यादित काळाची ऑफर! कमीत कमी कागदपत्रांसह Rs.{amount} पर्यंतचे इन्स्टंट पर्सनल लोन मिळवा. आजच अर्ज करा.",
            "दर आठवड्याला रोमांचक बक्षिसे जिंका! सहभागी होण्यासाठी WIN रिप्लाय करा. अटी लागू.",
            "{telecom} च्या Rs.{amount} रिचार्जसोबत मिळवा 2 मोफत मूव्ही तिकिटे, फक्त याच आठवड्यात.",
            "फक्त Rs.{amount}/महिना मध्ये खास इन्शुरन्स प्लान. तुमच्या कुटुंबाचे संरक्षण करा. कॉल करा {phone}.",
            "घरबसल्या रोज Rs.{amount} कमवा. कोणतीही गुंतवणूक नाही. सुरू करण्यासाठी YES रिप्लाय करा.",
            "{company} वर फ्लॅट 70% सूट - फक्त आजच! चेकआउटवर SAVE70 कोड वापरा.",
            "{city} मध्ये सिंगल्स तुमच्याशी चॅट करू इच्छितात! आत्ताच NOW रिप्लाय करून जोडा.",
            "झिरो वार्षिक शुल्क असलेले मोफत क्रेडिट कार्ड मिळवा. आत्ताच अर्ज करा आणि Rs.{amount} वेलकम बोनस मिळवा.",
            "तुमचा {telecom} नंबर मोफत 5G अपग्रेडसाठी पात्र आहे. सक्रिय करण्यासाठी UPGRADE रिप्लाय करा.",
            "उत्तम फॉरेक्स ट्रेडिंग टिप्स! Rs.{amount} चे एका महिन्यात Rs.{amount2} करा. आत्ताच कॉल करा {phone}.",
            "या वीकेंडला फक्त {company} च्या सर्व उत्पादनांवर Buy 1 Get 1 Free. लवकर करा, ऑफर लवकरच संपेल!",
            "तुमचे कर्ज आजच संपवा! कमी व्याजदरात लोन एकत्र करा. कॉल करा {phone}.",
            "आमच्या खास डाएट प्लॅनने 30 दिवसांत 10 किलो वजन कमी करा. माहितीसाठी INFO रिप्लाय करा.",
            "तुमचे ज्योतिष वाचन सांगते मोठे बदल येणार आहेत! मोफत सल्ल्यासाठी कॉल करा {phone}.",
            "अभिनंदन! तुम्ही Rs.{amount} च्या प्री-अप्रूव्ह्ड लोनसाठी पात्र आहात. आत्ताच APPLY रिप्लाय करा.",
            "{company} वर न्यू इयर सेल! संपूर्ण स्टोअरवर फ्लॅट 60% सूट. स्टॉक संपण्याआधी खरेदी करा.",
            "{city} साठी तुमची स्वप्नातील सुट्टी अविश्वसनीय किमतीत बुक करा. पॅकेजसाठी कॉल करा {phone}.",
            "तुमच्या क्रेडिट कार्डसाठी इन्स्टंट अप्रूव्हल मिळवा, उत्पन्नाचा पुरावा लागत नाही. YES रिप्लाय करा.",
            "3 मित्रांना रेफर करून Rs.{amount} चा मोफत रिचार्ज मिळवा. आत्ताच REFER रिप्लाय करा.",
        ],
        "Smishing": [
            "प्रिय ग्राहक, तुमचे {bank} खाते आज ब्लॉक होईल. लगेच केवायसी अपडेट करा: {url}",
            "तुमचे {courier} पार्सल Rs.{amount} थकीत कस्टम शुल्कामुळे होल्डवर आहे. आत्ताच पेमेंट करा: {url}",
            "अभिनंदन! तुमचा मोबाईल नंबर {company} लकी ड्रॉमध्ये Rs.{amount} जिंकला आहे. आत्ताच दावा करा: {url}",
            "तातडीचे: तुमचे {bank} डेबिट कार्ड तात्पुरते निलंबित करण्यात आले आहे. इथे तुमची माहिती व्हेरिफाय करा: {url}",
            "तुमचे आधार कार्ड निष्क्रिय होईल. दंड टाळण्यासाठी लगेच {url} वर पॅनशी लिंक करा.",
            "Rs.{amount} चा इन्कम टॅक्स रिफंड प्रलंबित आहे. {date} पूर्वी दावा करण्यासाठी {url} वर क्लिक करा.",
            "थकीत बिलामुळे तुमचे वीज कनेक्शन आज रात्री तोडले जाईल. लगेच पेमेंट करा: {url}",
            "प्रिय वापरकर्ता, तुमची {wallet} केवायसी संपली आहे. आत्ताच अपडेट करा नाहीतर वॉलेट ब्लॉक होईल: {url}",
            "तुमची {company} मध्ये Rs.{amount}/महिना पगाराच्या नोकरीसाठी निवड झाली आहे. आत्ताच नोंदणी करा: {url}",
            "तुमचे पार्सल डिलिव्हर होऊ शकले नाही. Rs.{amount} शिपिंग शुल्क भरून पुन्हा शेड्यूल करा: {url}",
            "{bank} अलर्ट: तुमच्या खात्यात असामान्य लॉगिन आढळले आहे. आत्ताच सुरक्षित करा: {url}",
            "तुमचे {wallet} खाते तात्पुरते लॉक झाले आहे. अनलॉक करण्यासाठी तुमची ओळख व्हेरिफाय करा: {url}",
            "प्रिय अर्जदार, तुमचा {govt} दस्तऐवज तयार आहे. डाउनलोड करा आणि प्रोसेसिंग शुल्क भरा: {url}",
            "तुमचे सिम कार्ड 24 तासांत ब्लॉक होईल. आत्ताच तुमची {telecom} केवायसी अपडेट करा: {url}",
            "अंतिम सूचना: प्रलंबित व्हेरिफिकेशनमुळे तुमचे {bank} खाते निलंबित केले जाईल. क्लिक करा: {url}",
            "तुमचा क्रेडिट स्कोअर घसरला आहे! आत्ताच {url} वर तपासा आणि दुरुस्त करा.",
            "तुम्हाला {wallet} द्वारे Rs.{amount} चे पेमेंट मिळाले आहे. स्वीकारण्यासाठी इथे क्लिक करा: {url}",
            "प्रिय ग्राहक, तुमची वार्षिक बँक लॉकर केवायसी पूर्ण करा नाहीतर तो सील केला जाईल: {url}",
            "तुमच्या {company} खात्यात संशयास्पद क्रियाकलाप दिसत आहे. कायमस्वरूपी निलंबन टाळण्यासाठी आत्ताच व्हेरिफाय करा: {url}",
            "अभिनंदन {name}! तुम्ही {company} कडून iPhone जिंकला आहे. तुमचे बक्षीस मिळवा: {url}",
        ],
    },
}

LANG_FILES = {
    "english": "Synthetic_English.csv",
    "hinglish": "Synthetic_Hinglish.csv",
    "hindi": "Synthetic_Hindi.csv",
    "marathi": "Synthetic_Marathi.csv",
}


def generate_language_file(lang, total_target, out_path):
    templates = TEMPLATES[lang]
    names = NAMES[lang]
    cities = CITIES[lang]
    rows = []
    seen = set()
    per_label = total_target // len(LABELS)

    for label in LABELS:
        tmpl_list = templates[label]
        count = 0
        attempts = 0
        max_attempts = per_label * 50
        while count < per_label and attempts < max_attempts:
            attempts += 1
            template = random.choice(tmpl_list)
            has_url = "{url}" in template
            has_phone = "{phone}" in template
            text = fill(template, names, cities)

            email_flag = "No"
            if random.random() < 0.02:
                text = text + f" Email: {rand_email()}"
                email_flag = "yes"

            if text in seen:
                continue
            seen.add(text)

            rows.append({
                "LABEL": label,
                "TEXT": text,
                "URL": "yes" if has_url else "No",
                "EMAIL": email_flag,
                "PHONE": "yes" if has_phone else "No",
            })
            count += 1

    random.shuffle(rows)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["LABEL", "TEXT", "URL", "EMAIL", "PHONE"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for lang, fname in LANG_FILES.items():
        out_path = os.path.join(OUT_DIR, fname)
        n = generate_language_file(lang, TOTAL_PER_LANG, out_path)
        total += n
        print(f"{lang}: {n} rows written to {out_path}")
    print(f"TOTAL rows across all languages: {total}")
