# -*- coding: utf-8 -*-
"""
Adds 'ham' (legitimate) SMS examples that legitimately contain URLs
(order tracking, meeting links, e-statements, etc. from real well-known
domains). This directly counters the risk of the model learning
"URL present => spam", since the base synthetic set only ever puts URLs
in Smishing templates.
"""
import csv
import os
import random
import string

random.seed(7)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset_5971")

BANKS = ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "Punjab National Bank",
         "Bank of Baroda", "Kotak Mahindra Bank", "Canara Bank"]
COMPANIES = ["Amazon", "Flipkart", "Myntra", "Meesho", "Ajio", "BigBasket", "Nykaa"]
COURIERS = ["Blue Dart", "Delhivery", "DTDC", "India Post", "Ekart", "FedEx", "Xpressbees"]
TELECOM = ["Airtel", "Jio", "Vi", "BSNL"]

LEGIT_DOMAINS = [
    "amazon.in/track", "flipkart.com/orders", "meet.google.com/abc-defg-hij",
    "zoom.us/j/1234567890", "maps.google.com/maps", "irctc.co.in/pnr",
    "youtube.com/watch", "en.wikipedia.org/wiki", "netflix.com/browse",
    "swiggy.com/order", "zomato.com/order", "linkedin.com/in/profile",
    "drive.google.com/file", "docs.google.com/document", "airtel.in/bill",
    "hdfcbank.com/statements", "myntra.com/orders", "bigbasket.com/order",
    "uber.com/trip", "olacabs.com/trip", "bookmyshow.com/booking",
    "practo.com/appointments", "policybazaar.com/documents", "redbus.in/booking",
    "livpure.com/service", "indigo.in/checkin", "1mg.com/track",
    "dineout.co.in/booking", "urbancompany.com/booking", "cleartrip.com/booking",
    "yatra.com/booking", "cred.club/statements", "groww.in/portfolio",
    "dominos.co.in/order", "makemytrip.com/booking", "spicejet.com/checkin",
    "webex.com/meet", "microsoft.com/teams", "github.com/notifications",
    "paytmbank.com/statements", "icicibank.com/statements", "axisbank.com/statements",
    "lenskart.com/order", "pharmeasy.in/order", "nykaa.com/order",
    "confirmtkt.com/pnr", "rapido.bike/trip", "porter.in/booking",
    "housing.com/property", "naukri.com/profile", "byjus.com/class",
    "upstox.com/portfolio", "zerodha.com/reports", "policyx.com/renew",
]

NAMES = {
    "english": ["Raj", "Priya", "Amit", "Sneha", "Rahul", "Anita", "Vikram", "Pooja"],
    "hinglish": ["Raj", "Priya", "Amit", "Sneha", "Rahul", "Anita", "Vikram", "Pooja"],
    "hindi": ["राज", "प्रिया", "अमित", "स्नेहा", "राहुल", "अनीता", "विक्रम", "पूजा"],
    "marathi": ["राज", "प्रिया", "अमोल", "स्नेहल", "राहुल", "अनिता", "विक्रम", "पूजा"],
}


def rand_amount():
    value = random.choice([random.randint(50, 999), random.randint(1000, 9999), random.randint(10000, 99999)])
    return f"{value:,}"


def rand_orderid():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))


def rand_time():
    hour = random.randint(1, 12)
    minute = random.randint(0, 59)
    ap = random.choice(["AM", "PM"])
    return f"{hour}:{minute:02d} {ap}"


def rand_month():
    return random.choice(["January", "February", "March", "April", "May", "June",
                          "July", "August", "September", "October", "November", "December"])


def rand_legit_url():
    domain = random.choice(LEGIT_DOMAINS)
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    return f"https://{domain}/{suffix}"


def fill(template, names):
    return template.format(
        bank=random.choice(BANKS),
        company=random.choice(COMPANIES),
        courier=random.choice(COURIERS),
        telecom=random.choice(TELECOM),
        name=random.choice(names),
        amount=rand_amount(),
        orderid=rand_orderid(),
        time=rand_time(),
        month=rand_month(),
        legit_url=rand_legit_url(),
    )


TEMPLATES = {
    "english": [
        "Your {courier} order has shipped! Track it live here: {legit_url}",
        "Your {company} order #{orderid} is out for delivery. Track: {legit_url}",
        "Here's the location for tonight's dinner: {legit_url}",
        "Join our team meeting at {time}: {legit_url}",
        "Your flight ticket is confirmed. View e-ticket: {legit_url}",
        "Payment of Rs.{amount} received. View receipt: {legit_url}",
        "Check out this article I found interesting: {legit_url}",
        "Your {telecom} bill is ready to view online: {legit_url}",
        "Here is the recipe I mentioned: {legit_url}",
        "Your train PNR {orderid} status: {legit_url}",
        "Class notes uploaded, download here: {legit_url}",
        "Watch this video, it's hilarious: {legit_url}",
        "Your {bank} e-statement for {month} is available: {legit_url}",
        "RSVP for the wedding here: {legit_url}",
        "Your {company} return has been approved. Track refund: {legit_url}",
    ],
    "hinglish": [
        "Aapka {courier} order ship ho gaya hai! Yahan live track karein: {legit_url}",
        "Aapka {company} order #{orderid} out for delivery hai. Track karein: {legit_url}",
        "Aaj raat ke dinner ka location yeh hai: {legit_url}",
        "{time} baje team meeting join karein: {legit_url}",
        "Aapki flight ticket confirm ho gayi hai. E-ticket dekhein: {legit_url}",
        "Rs.{amount} ka payment mil gaya hai. Receipt dekhein: {legit_url}",
        "Yeh interesting article dekho jo maine dhoonda: {legit_url}",
        "Aapka {telecom} bill online dekhne ke liye ready hai: {legit_url}",
        "Woh recipe jo maine bataya tha, yeh raha: {legit_url}",
        "Aapke train PNR {orderid} ka status: {legit_url}",
        "Class ke notes upload ho gaye hain, yahan se download karein: {legit_url}",
        "Yeh video dekho, bahut funny hai: {legit_url}",
        "Aapka {bank} e-statement {month} ke liye available hai: {legit_url}",
        "Shaadi ke liye yahan RSVP karein: {legit_url}",
        "Aapka {company} return approve ho gaya hai. Refund track karein: {legit_url}",
    ],
    "hindi": [
        "आपका {courier} ऑर्डर शिप हो गया है! इसे यहां लाइव ट्रैक करें: {legit_url}",
        "आपका {company} ऑर्डर #{orderid} डिलीवरी के लिए निकल चुका है। ट्रैक करें: {legit_url}",
        "आज रात के डिनर की लोकेशन यह है: {legit_url}",
        "{time} बजे टीम मीटिंग जॉइन करें: {legit_url}",
        "आपकी फ्लाइट टिकट कन्फर्म हो गई है। ई-टिकट देखें: {legit_url}",
        "Rs.{amount} का भुगतान प्राप्त हुआ। रसीद देखें: {legit_url}",
        "यह दिलचस्प लेख देखें जो मुझे मिला: {legit_url}",
        "आपका {telecom} बिल ऑनलाइन देखने के लिए तैयार है: {legit_url}",
        "वह रेसिपी जो मैंने बताई थी, यह रही: {legit_url}",
        "आपके ट्रेन पीएनआर {orderid} की स्थिति: {legit_url}",
        "क्लास के नोट्स अपलोड हो गए हैं, यहां से डाउनलोड करें: {legit_url}",
        "यह वीडियो देखो, बहुत मजेदार है: {legit_url}",
        "आपका {bank} ई-स्टेटमेंट {month} के लिए उपलब्ध है: {legit_url}",
        "शादी के लिए यहां RSVP करें: {legit_url}",
        "आपका {company} रिटर्न स्वीकृत हो गया है। रिफंड ट्रैक करें: {legit_url}",
    ],
    "marathi": [
        "तुमची {courier} ऑर्डर शिप झाली आहे! इथे लाइव्ह ट्रॅक करा: {legit_url}",
        "तुमची {company} ऑर्डर #{orderid} डिलिव्हरीसाठी निघाली आहे. ट्रॅक करा: {legit_url}",
        "आजच्या रात्रीच्या जेवणाचे ठिकाण हे आहे: {legit_url}",
        "{time} वाजता टीम मीटिंगमध्ये सामील व्हा: {legit_url}",
        "तुमचे फ्लाइट तिकीट कन्फर्म झाले आहे. ई-तिकीट पहा: {legit_url}",
        "Rs.{amount} चे पेमेंट मिळाले आहे. पावती पहा: {legit_url}",
        "मला सापडलेला हा मनोरंजक लेख पहा: {legit_url}",
        "तुमचे {telecom} बिल ऑनलाइन पाहण्यासाठी तयार आहे: {legit_url}",
        "मी सांगितलेली रेसिपी ही आहे: {legit_url}",
        "तुमच्या ट्रेन पीएनआर {orderid} ची स्थिती: {legit_url}",
        "वर्गाच्या नोट्स अपलोड झाल्या आहेत, इथून डाउनलोड करा: {legit_url}",
        "हा व्हिडिओ बघ, खूप मजेदार आहे: {legit_url}",
        "तुमचे {bank} ई-स्टेटमेंट {month} साठी उपलब्ध आहे: {legit_url}",
        "लग्नासाठी इथे RSVP करा: {legit_url}",
        "तुमचा {company} रिटर्न मंजूर झाला आहे. रिफंड ट्रॅक करा: {legit_url}",
    ],
}

PER_LANG_TARGET = 8000

if __name__ == "__main__":
    summary = []
    for lang, tmpl_list in TEMPLATES.items():
        names = NAMES[lang]
        rows = []
        seen = set()
        attempts = 0
        max_attempts = PER_LANG_TARGET * 30
        while len(rows) < PER_LANG_TARGET and attempts < max_attempts:
            attempts += 1
            template = random.choice(tmpl_list)
            text = fill(template, names)
            if text in seen:
                continue
            seen.add(text)
            rows.append({"LABEL": "ham", "TEXT": text, "URL": "yes", "EMAIL": "No", "PHONE": "No"})

        out_path = os.path.join(OUT_DIR, f"Synthetic_Ham_URL_{lang.capitalize()}.csv")
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["LABEL", "TEXT", "URL", "EMAIL", "PHONE"])
            writer.writeheader()
            writer.writerows(rows)
        summary.append(f"{lang}: {len(rows)} rows -> {out_path}")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "augment_ham_url_result.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
