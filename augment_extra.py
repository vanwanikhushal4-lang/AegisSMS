# -*- coding: utf-8 -*-
"""
Second-iteration data augmentation, added after diverse-test-set analysis
showed two gaps:
  1. Ham messages with civic/casual phrasing ("library books due", "let's
     plan a trip") were sparse, so the model over-associated words like
     "due"/"fee"/"trip" with spam.
  2. "Soft" promotional spam (discount codes, cashback, no urgency language)
     was under-represented, so the model leaned ham on wording-mild spam.
"""
import csv
import os
import random
import string

random.seed(11)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset_5971")

CITIES = {
    "english": ["Goa", "Manali", "Jaipur", "Udaipur", "Kerala", "Shimla", "Rishikesh", "Pondicherry"],
    "hinglish": ["Goa", "Manali", "Jaipur", "Udaipur", "Kerala", "Shimla", "Rishikesh", "Pondicherry"],
    "hindi": ["गोवा", "मनाली", "जयपुर", "उदयपुर", "केरल", "शिमला", "ऋषिकेश", "पांडिचेरी"],
    "marathi": ["गोवा", "मनाली", "जयपूर", "उदयपूर", "केरळ", "शिमला", "ऋषिकेश", "पॉंडिचेरी"],
}


def rand_date():
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.choice([2025, 2026])
    return f"{day:02d}-{month:02d}-{year}"


def rand_pct():
    return str(random.choice([10, 15, 20, 25, 30, 40, 50]))


def rand_code():
    return "".join(random.choices(string.ascii_uppercase, k=4)) + str(random.randint(10, 99))


def rand_amount():
    return f"{random.choice([99, 199, 299, 499, 999, 1499]):,}"


def fill(template, lang):
    return template.format(
        date=rand_date(), date2=rand_date(), city=random.choice(CITIES[lang]),
        pct=rand_pct(), code=rand_code(), amount=rand_amount(),
    )


HAM_TEMPLATES = {
    "english": [
        "If you haven't gotten your vaccine dose yet, please visit your nearest health centre.",
        "Reminder: your library books are due for return by {date}, late fee applies after.",
        "The municipal water supply will be interrupted tomorrow from 10 AM to 2 PM for maintenance.",
        "Voting for the society elections will be held this Sunday at the community hall.",
        "Blood donation camp organized at the community centre this weekend, all are welcome.",
        "The park will be closed for renovation from {date} to {date2}.",
        "Please segregate your waste into wet and dry bins as per municipal guidelines.",
        "The society AGM is scheduled for {date} at 6 PM in the clubhouse.",
        "Free health checkup camp at the local clinic this Saturday, walk-ins welcome.",
        "Reminder: submit your society maintenance payment by {date} to avoid late fee.",
        "Let's plan a trip to {city} next month, everyone free?",
        "Who's up for a weekend trip to {city}? Let me know by Friday.",
        "Thinking of a road trip to {city} this long weekend, you in?",
        "Found cheap flights to {city} for next month, should we book?",
        "Family trip to {city} is confirmed for December, so excited!",
        "Let's finally plan that {city} vacation we keep talking about.",
    ],
    "hinglish": [
        "Agar aapne abhi tak vaccine nahi lagwaya hai to nazdeeki health centre jaayein.",
        "Reminder: library ki kitabein {date} tak wapas karni hain, uske baad late fee lagegi.",
        "Kal subah 10 se 2 baje tak municipal paani ki supply band rahegi maintenance ke liye.",
        "Society elections ke liye voting is Sunday community hall mein hogi.",
        "Is weekend community centre mein blood donation camp hai, sab aa sakte hain.",
        "Park {date} se {date2} tak renovation ke liye band rahega.",
        "Kripya apna kachra geela aur sookha alag alag bins mein dalein.",
        "Society ki AGM {date} ko 6 baje clubhouse mein hai.",
        "Is Saturday local clinic mein free health checkup camp hai, sabka swagat hai.",
        "Reminder: society maintenance {date} tak jama karein warna late fee lagega.",
        "Agle mahine {city} trip plan karte hain, sab free ho na?",
        "Weekend trip {city} chalein kya? Friday tak batao.",
        "Is long weekend {city} road trip ka plan hai, chalega?",
        "{city} ke liye sasti flights mili hain, book kar lein?",
        "December mein {city} family trip confirm ho gaya hai, bahut excited hoon!",
        "Woh {city} vacation ab finally plan kar lete hain jiske baare mein baat kar rahe the.",
    ],
    "hindi": [
        "अगर आपने अभी तक अपना टीका नहीं लगवाया है तो नजदीकी केंद्र पर जाकर लगवाएं।",
        "अनुस्मारक: लाइब्रेरी की किताबें {date} तक वापस करनी हैं, उसके बाद विलंब शुल्क लगेगा।",
        "कल सुबह 10 से 2 बजे तक नगरपालिका जल आपूर्ति रखरखाव के लिए बंद रहेगी।",
        "सोसाइटी चुनाव के लिए मतदान इस रविवार को सामुदायिक भवन में होगा।",
        "इस सप्ताहांत सामुदायिक केंद्र में रक्तदान शिविर है, सभी आमंत्रित हैं।",
        "पार्क {date} से {date2} तक नवीनीकरण के लिए बंद रहेगा।",
        "कृपया अपने कचरे को गीले और सूखे डिब्बे में अलग करें।",
        "सोसाइटी की वार्षिक बैठक {date} को शाम 6 बजे क्लबहाउस में है।",
        "इस शनिवार स्थानीय क्लिनिक में मुफ्त स्वास्थ्य जांच शिविर है।",
        "अनुस्मारक: सोसाइटी का रखरखाव शुल्क {date} तक जमा करें अन्यथा विलंब शुल्क लगेगा।",
        "अगले महीने {city} की यात्रा प्लान करते हैं, सब फ्री हो ना?",
        "इस वीकेंड {city} चलें क्या? शुक्रवार तक बता देना।",
        "इस लॉन्ग वीकेंड {city} रोड ट्रिप का प्लान है, चलोगे?",
        "{city} के लिए सस्ती फ्लाइट मिली है, बुक कर लें?",
        "दिसंबर में {city} फैमिली ट्रिप कन्फर्म हो गई है, बहुत उत्साहित हूं!",
        "वो {city} वेकेशन अब आखिरकार प्लान कर लेते हैं जिसकी बात कर रहे थे।",
    ],
    "marathi": [
        "जर तुम्ही अजून लस घेतली नसेल तर जवळच्या केंद्रावर जाऊन घ्या.",
        "स्मरणपत्र: लायब्ररीची पुस्तके {date} पर्यंत परत करायची आहेत, त्यानंतर विलंब शुल्क लागेल.",
        "उद्या सकाळी 10 ते 2 वाजेपर्यंत महानगरपालिका पाणीपुरवठा देखभालीसाठी बंद राहील.",
        "सोसायटी निवडणुकीसाठी मतदान या रविवारी सामुदायिक सभागृहात होईल.",
        "या वीकेंडला सामुदायिक केंद्रात रक्तदान शिबिर आहे, सर्वांचे स्वागत आहे.",
        "उद्यान {date} ते {date2} पर्यंत नूतनीकरणासाठी बंद राहील.",
        "कृपया तुमचा कचरा ओला आणि सुका वेगवेगळ्या डब्यात टाका.",
        "सोसायटीची वार्षिक सभा {date} रोजी संध्याकाळी 6 वाजता क्लबहाऊसमध्ये आहे.",
        "या शनिवारी स्थानिक दवाखान्यात मोफत आरोग्य तपासणी शिबिर आहे.",
        "स्मरणपत्र: सोसायटीची देखभाल फी {date} पर्यंत भरा अन्यथा विलंब शुल्क लागेल.",
        "पुढच्या महिन्यात {city} ट्रिप प्लान करूया, सगळे फ्री आहात का?",
        "या वीकेंडला {city} जाऊया का? शुक्रवारपर्यंत सांगा.",
        "या लाँग वीकेंडला {city} रोड ट्रिपचा प्लान आहे, येणार का?",
        "{city} साठी स्वस्त फ्लाइट्स मिळाल्या आहेत, बुक करूया का?",
        "डिसेंबरमध्ये {city} फॅमिली ट्रिप कन्फर्म झाली आहे, खूप एक्साइटेड आहे!",
        "ती {city} सुट्टी आता शेवटी प्लान करूया ज्याबद्दल बोलत होतो.",
    ],
}

SPAM_TEMPLATES = {
    "english": [
        "Get flat {pct}% cashback on your first order using code {code} at checkout, valid till midnight.",
        "New arrivals are here! Shop the latest collection and save {pct}% today.",
        "Your favorite store just dropped a new sale. Use code {code} for {pct}% off.",
        "Enjoy {pct}% off on all orders above Rs.{amount} this weekend only.",
        "Refer a friend and both of you get Rs.{amount} off your next purchase.",
        "Members get early access to our {pct}% off sale starting today.",
        "Try our new menu and get a free dessert with code {code}.",
        "Subscribe now and get your first month free, cancel anytime.",
        "Upgrade your plan today and enjoy {pct}% off for 3 months.",
        "Shop the sale before it ends, extra {pct}% off with code {code}.",
    ],
    "hinglish": [
        "Pehle order par {pct}% cashback paayein code {code} use karke checkout par, aaj raat tak valid.",
        "New arrivals aa gaye hain! Latest collection shop karein aur {pct}% bachayein aaj.",
        "Aapke pasandida store par naya sale aaya hai. Code {code} se {pct}% off paayein.",
        "Is weekend Rs.{amount} se upar ke sabhi orders par {pct}% off paayein.",
        "Dost ko refer karein aur dono ko Rs.{amount} off milega agli purchase par.",
        "Members ko aaj se {pct}% off sale ka early access milega.",
        "Naya menu try karein aur code {code} se free dessert paayein.",
        "Abhi subscribe karein aur pehla mahina free paayein, kabhi bhi cancel karein.",
        "Apna plan aaj hi upgrade karein aur 3 mahine {pct}% off paayein.",
        "Sale khatam hone se pehle shop karein, code {code} se extra {pct}% off.",
    ],
    "hindi": [
        "पहले ऑर्डर पर {pct}% कैशबैक पाएं कोड {code} का उपयोग करके, आज रात तक वैध।",
        "नए आइटम आ गए हैं! नवीनतम कलेक्शन खरीदें और आज {pct}% बचाएं।",
        "आपके पसंदीदा स्टोर पर नया सेल आया है। कोड {code} से {pct}% छूट पाएं।",
        "इस वीकेंड Rs.{amount} से ऊपर के सभी ऑर्डर पर {pct}% छूट पाएं।",
        "दोस्त को रेफर करें और दोनों को अगली खरीद पर Rs.{amount} की छूट मिलेगी।",
        "मेंबर्स को आज से {pct}% छूट सेल का जल्दी एक्सेस मिलेगा।",
        "नया मेन्यू ट्राई करें और कोड {code} से मुफ्त डेजर्ट पाएं।",
        "अभी सब्सक्राइब करें और पहला महीना मुफ्त पाएं, कभी भी रद्द करें।",
        "अपना प्लान आज ही अपग्रेड करें और 3 महीने {pct}% छूट पाएं।",
        "सेल खत्म होने से पहले खरीदारी करें, कोड {code} से अतिरिक्त {pct}% छूट।",
    ],
    "marathi": [
        "पहिल्या ऑर्डरवर {pct}% कॅशबॅक मिळवा कोड {code} वापरून, आज मध्यरात्रीपर्यंत वैध.",
        "नवीन वस्तू आल्या आहेत! नवीनतम कलेक्शन खरेदी करा आणि आज {pct}% वाचवा.",
        "तुमच्या आवडत्या स्टोअरवर नवीन सेल आला आहे. कोड {code} ने {pct}% सूट मिळवा.",
        "या वीकेंडला Rs.{amount} वरील सर्व ऑर्डरवर {pct}% सूट मिळवा.",
        "मित्राला रेफर करा आणि दोघांनाही पुढच्या खरेदीवर Rs.{amount} सूट मिळेल.",
        "मेंबर्सना आजपासून {pct}% सूट सेलचा लवकर अॅक्सेस मिळेल.",
        "नवीन मेनू ट्राय करा आणि कोड {code} ने मोफत डेझर्ट मिळवा.",
        "आत्ताच सबस्क्राइब करा आणि पहिला महिना मोफत मिळवा, कधीही रद्द करा.",
        "तुमचा प्लॅन आजच अपग्रेड करा आणि 3 महिने {pct}% सूट मिळवा.",
        "सेल संपण्यापूर्वी खरेदी करा, कोड {code} ने अतिरिक्त {pct}% सूट.",
    ],
}

PER_LANG_TARGET = 4000

if __name__ == "__main__":
    summary = []
    for lang in ["english", "hinglish", "hindi", "marathi"]:
        for label, templates, fname in [
            ("ham", HAM_TEMPLATES[lang], f"Synthetic_Extra_Ham_{lang.capitalize()}.csv"),
            ("spam", SPAM_TEMPLATES[lang], f"Synthetic_Extra_SoftSpam_{lang.capitalize()}.csv"),
        ]:
            rows = []
            seen = set()
            attempts = 0
            max_attempts = PER_LANG_TARGET * 30
            while len(rows) < PER_LANG_TARGET and attempts < max_attempts:
                attempts += 1
                template = random.choice(templates)
                text = fill(template, lang)
                if text in seen:
                    continue
                seen.add(text)
                rows.append({"LABEL": label, "TEXT": text, "URL": "No", "EMAIL": "No", "PHONE": "No"})

            out_path = os.path.join(OUT_DIR, fname)
            with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=["LABEL", "TEXT", "URL", "EMAIL", "PHONE"])
                writer.writeheader()
                writer.writerows(rows)
            summary.append(f"{lang} {label}: {len(rows)} rows -> {out_path}")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "augment_extra_result.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
