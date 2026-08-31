# -*- coding: utf-8 -*-
"""
Third-iteration data augmentation, added after a user report that
intent-based fraud messages without URLs/phone numbers/classic urgency
keywords -- utility-termination threats, UPI "wrong transfer, please
refund" social engineering, and government-scheme/scholarship phishing
asking for bank/passbook details -- were being misclassified as Ham with
near-zero spam probability. The previous training data's Smishing examples
almost always paired a threat with a URL; these narrative patterns need
their own paraphrase diversity so the text embedding actually learns them,
not just a keyword flag.

Each spam category ships with matching ham counter-examples using similar
vocabulary (utility bills, scholarships, loans, personal payments, OTPs)
but legitimate framing, so the model learns the narrative pattern and the
"asks for credentials / wrong-transfer refund" framing -- not just "any
mention of scholarship/utility/refund is spam".
"""
import csv
import os
import random
import string

random.seed(23)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset_5971")

UTILITIES = ["Piped Gas", "Electricity", "Water", "LPG", "Broadband"]
SCHEMES = ["National Scholarship Portal", "PM Kisan Samman Nidhi", "PMAY",
           "Ayushman Bharat", "LPG Subsidy Scheme", "PM Ujjwala Yojana",
           "Post Matric Scholarship", "State Scholarship Scheme"]
BANKS = ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "Punjab National Bank",
         "Bank of Baroda", "Kotak Mahindra Bank", "Canara Bank"]
WALLETS = ["Google Pay", "PhonePe", "Paytm", "Amazon Pay"]
COMPANIES = ["Infosys", "TCS", "Reliance", "Flipkart", "Amazon", "HCL", "Wipro"]
UPI_HANDLES = ["okhdfcbank", "ybl", "oksbi", "ibl", "axl", "paytm", "okicici"]


def rand_amount():
    value = random.choice([random.randint(500, 999), random.randint(1000, 9999),
                           random.randint(10000, 99999)])
    return f"{value:,}"


def rand_upiid():
    name = ''.join(random.choices(string.ascii_lowercase, k=random.randint(5, 8)))
    if random.random() < 0.5:
        name += str(random.randint(10, 99))
    return f"{name}@{random.choice(UPI_HANDLES)}"


def rand_otp():
    length = random.choice([4, 6])
    return ''.join(random.choices(string.digits, k=length))


def rand_date():
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.choice([2025, 2026])
    return f"{day:02d}-{month:02d}-{year}"


def fill(template, lang):
    return template.format(
        utility=random.choice(UTILITIES),
        scheme=random.choice(SCHEMES),
        bank=random.choice(BANKS),
        wallet=random.choice(WALLETS),
        company=random.choice(COMPANIES),
        amount=rand_amount(),
        upiid=rand_upiid(),
        otp=rand_otp(),
        date=rand_date(),
    )


SPAM_TEMPLATES = {
    "english": [
        "{utility} connection will be terminated tonight due to pending meter verification.",
        "Your {utility} service will be discontinued tomorrow morning, verification pending at our end.",
        "Notice: your {utility} supply will be cut off due to incomplete meter reading update.",
        "Your {utility} meter inspection is overdue, service will be terminated within 24 hours.",
        "Final notice: {utility} connection at your address will be disconnected due to unverified meter details.",
        "Due to a technical error in your {utility} meter reading, your connection is scheduled for termination tonight.",
        "I accidentally sent Rs.{amount} to your {wallet} by mistake. Please refund it to this UPI ID: {upiid}",
        "Sorry, I transferred Rs.{amount} to the wrong number by mistake, this looks like your account. Kindly send it back to {upiid}.",
        "Hi, I think I sent Rs.{amount} to you by error while paying someone else. Please return it to UPI ID {upiid}, thank you.",
        "By mistake Rs.{amount} got transferred to your account from mine, please refund at the earliest to {upiid}.",
        "{scheme}: Your scholarship benefit of Rs.{amount} is ready for transfer. Submit your bank passbook details to receive it.",
        "{scheme} Alert: You are eligible for Rs.{amount} under the scheme. Share your Aadhaar and bank account number to claim.",
        "Congratulations, your application under {scheme} has been approved for Rs.{amount}. Send your bank details for disbursement.",
        "Your {bank} personal loan of Rs.{amount} is pre-approved. Share your bank account and Aadhaar details to process.",
        "Your insurance policy maturity amount of Rs.{amount} is ready. Send your bank passbook details to receive payment.",
        "You have been selected for a work-from-home job with {company}. Share your bank details to receive the joining bonus of Rs.{amount}.",
    ],
    "hinglish": [
        "{utility} connection aaj raat pending meter verification ki wajah se terminate ho jayega.",
        "Aapki {utility} service kal subah discontinue ho jayegi, verification hamari taraf se pending hai.",
        "Notice: aapki {utility} supply meter reading update incomplete hone ki wajah se cut off ho jayegi.",
        "Aapka {utility} meter inspection overdue hai, 24 ghante mein service terminate ho jayegi.",
        "Final notice: aapke address ka {utility} connection unverified meter details ki wajah se disconnect ho jayega.",
        "Aapke {utility} meter reading mein technical error ki wajah se, connection aaj raat termination ke liye scheduled hai.",
        "Maine galti se aapke {wallet} par Rs.{amount} bhej diya, kripya isse is UPI ID par refund kar dein: {upiid}",
        "Sorry, maine galti se galat number par Rs.{amount} transfer kar diya, yeh aapka account lagta hai. Kripya wapas bhej dein {upiid} par.",
        "Hi, mujhe lagta hai maine kisi aur ko pay karte waqt aapko galti se Rs.{amount} bhej diya. Kripya {upiid} par wapas bhej dein, dhanyavad.",
        "Galti se Rs.{amount} mere account se aapke account mein transfer ho gaye hain, jald se jald {upiid} par refund kar dein.",
        "{scheme}: Aapka scholarship benefit Rs.{amount} transfer ke liye ready hai. Ise paane ke liye apni bank passbook details submit karein.",
        "{scheme} Alert: Aap scheme ke tahat Rs.{amount} ke liye eligible hain. Claim karne ke liye apna Aadhaar aur bank account number share karein.",
        "Badhai ho, {scheme} ke tahat aapka application Rs.{amount} ke liye approve ho gaya hai. Disbursement ke liye apni bank details bhejein.",
        "Aapka {bank} personal loan Rs.{amount} ka pre-approved hai. Process karne ke liye apni bank account aur Aadhaar details share karein.",
        "Aapki insurance policy ka maturity amount Rs.{amount} ready hai. Payment paane ke liye apni bank passbook details bhejein.",
        "Aapko {company} ke saath work-from-home job ke liye select kiya gaya hai. Rs.{amount} ka joining bonus paane ke liye apni bank details share karein.",
    ],
    "hindi": [
        "{utility} कनेक्शन पेंडिंग मीटर सत्यापन के कारण आज रात समाप्त कर दिया जाएगा।",
        "आपकी {utility} सेवा कल सुबह बंद कर दी जाएगी, सत्यापन हमारी ओर से लंबित है।",
        "सूचना: मीटर रीडिंग अपडेट अधूरा होने के कारण आपकी {utility} आपूर्ति काट दी जाएगी।",
        "आपका {utility} मीटर निरीक्षण बाकी है, 24 घंटे में सेवा समाप्त कर दी जाएगी।",
        "अंतिम सूचना: असत्यापित मीटर विवरण के कारण आपके पते का {utility} कनेक्शन डिस्कनेक्ट कर दिया जाएगा।",
        "आपकी {utility} मीटर रीडिंग में तकनीकी त्रुटि के कारण, कनेक्शन आज रात समाप्ति के लिए निर्धारित है।",
        "मैंने गलती से आपके {wallet} पर Rs.{amount} भेज दिए, कृपया इसे इस UPI ID पर रिफंड करें: {upiid}",
        "क्षमा करें, मैंने गलती से गलत नंबर पर Rs.{amount} ट्रांसफर कर दिए, यह आपका खाता लगता है। कृपया {upiid} पर वापस भेजें।",
        "हाय, मुझे लगता है मैंने किसी और को भुगतान करते समय गलती से आपको Rs.{amount} भेज दिए। कृपया {upiid} पर वापस भेजें, धन्यवाद।",
        "गलती से Rs.{amount} मेरे खाते से आपके खाते में ट्रांसफर हो गए हैं, जल्द से जल्द {upiid} पर रिफंड करें।",
        "{scheme}: आपका छात्रवृत्ति लाभ Rs.{amount} ट्रांसफर के लिए तैयार है। इसे प्राप्त करने के लिए अपनी बैंक पासबुक विवरण जमा करें।",
        "{scheme} अलर्ट: आप योजना के तहत Rs.{amount} के लिए पात्र हैं। दावा करने के लिए अपना आधार और बैंक खाता नंबर साझा करें।",
        "बधाई हो, {scheme} के तहत आपका आवेदन Rs.{amount} के लिए स्वीकृत हो गया है। वितरण के लिए अपनी बैंक विवरण भेजें।",
        "आपका {bank} पर्सनल लोन Rs.{amount} का पूर्व-स्वीकृत है। प्रक्रिया के लिए अपना बैंक खाता और आधार विवरण साझा करें।",
        "आपकी बीमा पॉलिसी की परिपक्वता राशि Rs.{amount} तैयार है। भुगतान प्राप्त करने के लिए अपनी बैंक पासबुक विवरण भेजें।",
        "आपको {company} के साथ वर्क-फ्रॉम-होम जॉब के लिए चुना गया है। Rs.{amount} का जॉइनिंग बोनस पाने के लिए अपनी बैंक विवरण साझा करें।",
    ],
    "marathi": [
        "{utility} कनेक्शन प्रलंबित मीटर पडताळणीमुळे आज रात्री बंद केले जाईल.",
        "तुमची {utility} सेवा उद्या सकाळी बंद केली जाईल, आमच्याकडून पडताळणी प्रलंबित आहे.",
        "सूचना: मीटर रीडिंग अपडेट अपूर्ण असल्यामुळे तुमचा {utility} पुरवठा तोडला जाईल.",
        "तुमचे {utility} मीटर निरीक्षण बाकी आहे, 24 तासांत सेवा बंद केली जाईल.",
        "अंतिम सूचना: असत्यापित मीटर तपशीलामुळे तुमच्या पत्त्यावरील {utility} कनेक्शन डिस्कनेक्ट केले जाईल.",
        "तुमच्या {utility} मीटर रीडिंगमध्ये तांत्रिक त्रुटीमुळे, कनेक्शन आज रात्री बंद करण्यासाठी नियोजित आहे.",
        "मी चुकून तुमच्या {wallet} वर Rs.{amount} पाठवले, कृपया हे या UPI ID वर परत करा: {upiid}",
        "माफ करा, मी चुकून चुकीच्या नंबरवर Rs.{amount} ट्रान्सफर केले, हे तुमचे खाते वाटते. कृपया {upiid} वर परत पाठवा.",
        "हाय, मला वाटते मी दुसऱ्याला पेमेंट करताना चुकून तुम्हाला Rs.{amount} पाठवले. कृपया {upiid} वर परत पाठवा, धन्यवाद.",
        "चुकून माझ्या खात्यातून तुमच्या खात्यात Rs.{amount} ट्रान्सफर झाले आहेत, लवकरात लवकर {upiid} वर रिफंड करा.",
        "{scheme}: तुमचा शिष्यवृत्ती लाभ Rs.{amount} ट्रान्सफरसाठी तयार आहे. मिळवण्यासाठी तुमची बँक पासबुक माहिती सबमिट करा.",
        "{scheme} अलर्ट: तुम्ही योजनेअंतर्गत Rs.{amount} साठी पात्र आहात. दावा करण्यासाठी तुमचा आधार आणि बँक खाते क्रमांक शेअर करा.",
        "अभिनंदन, {scheme} अंतर्गत तुमचा अर्ज Rs.{amount} साठी मंजूर झाला आहे. वितरणासाठी तुमची बँक माहिती पाठवा.",
        "तुमचे {bank} पर्सनल लोन Rs.{amount} चे पूर्व-मंजूर आहे. प्रक्रियेसाठी तुमचे बँक खाते आणि आधार तपशील शेअर करा.",
        "तुमच्या विमा पॉलिसीची मुदतपूर्ती रक्कम Rs.{amount} तयार आहे. पेमेंट मिळवण्यासाठी तुमची बँक पासबुक माहिती पाठवा.",
        "तुम्हाला {company} सोबत वर्क-फ्रॉम-होम जॉबसाठी निवडले आहे. Rs.{amount} चा जॉइनिंग बोनस मिळवण्यासाठी तुमची बँक माहिती शेअर करा.",
    ],
}

HAM_TEMPLATES = {
    "english": [
        "Your {utility} bill of Rs.{amount} has been generated for this month, due by {date}.",
        "Your {utility} meter reading has been updated successfully, no action needed.",
        "Thank you, your {utility} payment of Rs.{amount} was received and your connection is active.",
        "Your {scheme} application has been received successfully, status can be checked on the official portal.",
        "Your scholarship of Rs.{amount} under {scheme} has been credited to your registered bank account.",
        "Your loan EMI of Rs.{amount} was auto-debited successfully today.",
        "Your insurance premium payment of Rs.{amount} was successful, thank you.",
        "Sent you Rs.{amount} for the dinner split, let me know once you get it.",
        "Your OTP is {otp}, do not share it with anyone, not even bank staff.",
        "Your {utility} connection has passed the routine meter inspection, no issues found.",
        "Your job application with {company} has been received, HR will contact you within a week.",
        "Your {bank} loan account statement for this month is now available in the app.",
    ],
    "hinglish": [
        "Aapka {utility} bill Rs.{amount} ka is mahine generate hua hai, {date} tak due hai.",
        "Aapki {utility} meter reading successfully update ho gayi hai, koi action nahi chahiye.",
        "Dhanyavad, aapka {utility} payment Rs.{amount} ka receive ho gaya hai aur connection active hai.",
        "Aapka {scheme} application successfully receive ho gaya hai, status official portal par check kar sakte hain.",
        "Aapki {scheme} ke tahat Rs.{amount} ki scholarship aapke registered bank account mein credit ho gayi hai.",
        "Aapka loan EMI Rs.{amount} ka aaj successfully auto-debit ho gaya.",
        "Aapka insurance premium payment Rs.{amount} ka successful raha, dhanyavad.",
        "Dinner split ke liye Rs.{amount} bhej diye hain, mil jaye to batana.",
        "Aapka OTP {otp} hai, kisi ke saath share mat karna, bank staff ke saath bhi nahi.",
        "Aapke {utility} connection ka routine meter inspection ho gaya hai, koi issue nahi mila.",
        "{company} ke saath aapki job application receive ho gayi hai, HR ek hafte mein contact karega.",
        "Aapke {bank} loan account ka is mahine ka statement app mein available hai.",
    ],
    "hindi": [
        "आपका {utility} बिल Rs.{amount} का इस महीने जनरेट हुआ है, {date} तक देय है।",
        "आपकी {utility} मीटर रीडिंग सफलतापूर्वक अपडेट हो गई है, कोई कार्रवाई आवश्यक नहीं है।",
        "धन्यवाद, आपका {utility} भुगतान Rs.{amount} का प्राप्त हो गया है और कनेक्शन सक्रिय है।",
        "आपका {scheme} आवेदन सफलतापूर्वक प्राप्त हो गया है, स्थिति आधिकारिक पोर्टल पर देखी जा सकती है।",
        "आपकी {scheme} के तहत Rs.{amount} की छात्रवृत्ति आपके पंजीकृत बैंक खाते में जमा हो गई है।",
        "आपका लोन EMI Rs.{amount} का आज सफलतापूर्वक ऑटो-डेबिट हो गया।",
        "आपका बीमा प्रीमियम भुगतान Rs.{amount} का सफल रहा, धन्यवाद।",
        "डिनर स्प्लिट के लिए Rs.{amount} भेज दिए हैं, मिल जाए तो बताना।",
        "आपका OTP {otp} है, इसे किसी के साथ साझा न करें, बैंक स्टाफ के साथ भी नहीं।",
        "आपके {utility} कनेक्शन का नियमित मीटर निरीक्षण हो गया है, कोई समस्या नहीं मिली।",
        "{company} के साथ आपका जॉब आवेदन प्राप्त हो गया है, HR एक सप्ताह में संपर्क करेगा।",
        "आपके {bank} लोन खाते का इस महीने का स्टेटमेंट ऐप में उपलब्ध है।",
    ],
    "marathi": [
        "तुमचे {utility} बिल Rs.{amount} चे या महिन्यात तयार झाले आहे, {date} पर्यंत देय आहे.",
        "तुमची {utility} मीटर रीडिंग यशस्वीरित्या अपडेट झाली आहे, कोणतीही कृती आवश्यक नाही.",
        "धन्यवाद, तुमचे {utility} पेमेंट Rs.{amount} चे प्राप्त झाले आहे आणि कनेक्शन सक्रिय आहे.",
        "तुमचा {scheme} अर्ज यशस्वीरित्या प्राप्त झाला आहे, स्थिती अधिकृत पोर्टलवर तपासता येईल.",
        "तुमची {scheme} अंतर्गत Rs.{amount} ची शिष्यवृत्ती तुमच्या नोंदणीकृत बँक खात्यात जमा झाली आहे.",
        "तुमचा लोन EMI Rs.{amount} चा आज यशस्वीरित्या ऑटो-डेबिट झाला.",
        "तुमचे विमा प्रीमियम पेमेंट Rs.{amount} चे यशस्वी झाले, धन्यवाद.",
        "डिनर स्प्लिटसाठी Rs.{amount} पाठवले आहेत, मिळाले की सांग.",
        "तुमचा OTP {otp} आहे, तो कोणाशीही शेअर करू नका, बँक कर्मचाऱ्यांशीही नाही.",
        "तुमच्या {utility} कनेक्शनची नियमित मीटर तपासणी झाली आहे, कोणतीही समस्या आढळली नाही.",
        "{company} सोबत तुमचा जॉब अर्ज प्राप्त झाला आहे, HR एका आठवड्यात संपर्क करेल.",
        "तुमच्या {bank} लोन खात्याचे या महिन्याचे स्टेटमेंट अ‍ॅपमध्ये उपलब्ध आहे.",
    ],
}

PER_LANG_TARGET = 5000

if __name__ == "__main__":
    summary = []
    for lang in ["english", "hinglish", "hindi", "marathi"]:
        for label, templates, fname in [
            ("spam", SPAM_TEMPLATES[lang], f"Synthetic_Fraud_Spam_{lang.capitalize()}.csv"),
            ("ham", HAM_TEMPLATES[lang], f"Synthetic_Fraud_Ham_{lang.capitalize()}.csv"),
        ]:
            out_rows = []
            seen = set()
            attempts = 0
            max_attempts = PER_LANG_TARGET * 40
            while len(out_rows) < PER_LANG_TARGET and attempts < max_attempts:
                attempts += 1
                template = random.choice(templates)
                text = fill(template, lang)
                if text in seen:
                    continue
                seen.add(text)
                out_rows.append({"LABEL": label, "TEXT": text, "URL": "No", "EMAIL": "No", "PHONE": "No"})

            out_path = os.path.join(OUT_DIR, fname)
            with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=["LABEL", "TEXT", "URL", "EMAIL", "PHONE"])
                writer.writeheader()
                writer.writerows(out_rows)
            summary.append(f"{lang} {label}: {len(out_rows)} rows -> {out_path}")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "augment_fraud_patterns_result.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
