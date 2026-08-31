# -*- coding: utf-8 -*-
"""
Fourth-iteration data augmentation, closing gaps found by challenge_test_set.csv:
  1. OTP/PIN "share to receive X" scams were never actually in training data
     (near-zero spam probability on all such examples).
  2. The refund-scam keyword ("accidentally sent"/"galti se"/"गलती से" etc.)
     was firing near-1.0 confidence on casual, friendly-toned Hindi/Marathi
     messages between friends -- because the only ham counter-example for
     that category ("sent you Rs.X for the dinner split") never actually
     used the "accidentally sent... please send back" phrasing itself, so
     the model never saw a genuinely friendly example of that exact framing
     to contrast against the scam framing (formal tone, unknown sender,
     explicit UPI ID). This batch adds several such ham examples per
     language, plus a bit more utility-termination / scheme-phishing
     paraphrase variety and more everyday personal-message diversity to
     counter a regression seen on diverse_test_set.csv after the last
     training round.
"""
import csv
import os
import random
import string

random.seed(31)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset_5971")

WALLETS = ["Google Pay", "PhonePe", "Paytm", "Amazon Pay"]
SCHEMES = ["National Scholarship Portal", "PM Kisan Samman Nidhi", "PMAY",
           "Ayushman Bharat", "LPG Subsidy Scheme", "PM Ujjwala Yojana",
           "Post Matric Scholarship", "State Scholarship Scheme",
           "PM Awas Yojana", "EPFO", "Ration Card Portal", "PM Fasal Bima Yojana"]
UTILITIES = ["Piped Gas", "Electricity", "Water", "LPG", "Broadband", "DTH"]
NAMES = {
    "english": ["Raj", "Priya", "Amit", "Sneha", "Rahul", "Anita", "Vikram", "Pooja"],
    "hinglish": ["Raj", "Priya", "Amit", "Sneha", "Rahul", "Anita", "Vikram", "Pooja"],
    "hindi": ["राज", "प्रिया", "अमित", "स्नेहा", "राहुल", "अनीता", "विक्रम", "पूजा"],
    "marathi": ["राज", "प्रिया", "अमोल", "स्नेहल", "राहुल", "अनिता", "विक्रम", "पूजा"],
}


def rand_amount():
    return f"{random.choice([random.randint(50, 999), random.randint(1000, 9999)]):,}"


def rand_otp():
    return ''.join(random.choices(string.digits, k=random.choice([4, 6])))


def fill(template, lang):
    return template.format(
        wallet=random.choice(WALLETS), scheme=random.choice(SCHEMES),
        utility=random.choice(UTILITIES), amount=rand_amount(),
        otp=rand_otp(), name=random.choice(NAMES[lang]),
    )


SPAM_TEMPLATES = {
    "english": [
        "To release your pending cashback of Rs.{amount}, please share the OTP you just received with our executive.",
        "Your refund is stuck, kindly share the verification code sent to your phone to complete the process.",
        "Your gas cylinder subsidy amount will be credited only after you share the OTP sent for verification.",
        "We need to verify your identity, please read out the OTP you received to our support agent on call.",
        "Your prize money will be released as soon as you share the 6-digit code sent to your mobile.",
        "{scheme}: to confirm your beneficiary status, share the OTP sent to your registered number with our helpline.",
        "Your electricity connection will be stopped this evening, non-payment recorded on our system.",
        "Your water service will be discontinued from tomorrow, outstanding dues detected on your meter.",
        "Your DTH recharge validation failed, connection will be cut off unless you verify within an hour.",
        "{scheme}: your application is on hold, respond with your bank passbook photo to proceed.",
    ],
    "hinglish": [
        "Aapka pending cashback Rs.{amount} release karne ke liye, abhi mila OTP hamare executive ke saath share karein.",
        "Aapka refund atka hua hai, process complete karne ke liye phone par aaya verification code share karein.",
        "Aapka gas cylinder subsidy amount tabhi credit hoga jab aap verification ke liye bheja gaya OTP share karenge.",
        "Aapki identity verify karni hai, kripya call par hamare support agent ko OTP bata dein.",
        "Aapki prize money tabhi release hogi jab aap mobile par mila 6-digit code share karenge.",
        "{scheme}: beneficiary status confirm karne ke liye, registered number par aaya OTP hamari helpline ko share karein.",
        "Aapka electricity connection aaj shaam band ho jayega, hamare system mein non-payment record hua hai.",
        "Aapki water service kal se band ho jayegi, aapke meter mein outstanding dues mile hain.",
        "Aapka DTH recharge validation fail ho gaya, 1 ghante mein verify na karne par connection cut ho jayega.",
        "{scheme}: aapka application hold par hai, aage badhne ke liye apni bank passbook photo bhejein.",
    ],
    "hindi": [
        "आपका पेंडिंग कैशबैक Rs.{amount} रिलीज़ करने के लिए, कृपया अभी मिला OTP हमारे एक्जीक्यूटिव के साथ साझा करें।",
        "आपका रिफंड अटका हुआ है, प्रक्रिया पूरी करने के लिए फोन पर आया वेरिफिकेशन कोड साझा करें।",
        "आपकी गैस सिलेंडर सब्सिडी राशि तभी क्रेडिट होगी जब आप सत्यापन के लिए भेजा गया OTP साझा करेंगे।",
        "आपकी पहचान सत्यापित करनी है, कृपया कॉल पर हमारे सपोर्ट एजेंट को OTP बताएं।",
        "आपकी पुरस्कार राशि तभी जारी होगी जब आप मोबाइल पर मिला 6-अंकों का कोड साझा करेंगे।",
        "{scheme}: लाभार्थी स्थिति की पुष्टि के लिए, पंजीकृत नंबर पर आया OTP हमारी हेल्पलाइन को साझा करें।",
        "आपका बिजली कनेक्शन आज शाम बंद कर दिया जाएगा, हमारे सिस्टम में भुगतान न होने का रिकॉर्ड है।",
        "आपकी पानी की सेवा कल से बंद कर दी जाएगी, आपके मीटर में बकाया राशि पाई गई है।",
        "आपका DTH रिचार्ज वेरिफिकेशन फेल हो गया, 1 घंटे में सत्यापित न करने पर कनेक्शन काट दिया जाएगा।",
        "{scheme}: आपका आवेदन होल्ड पर है, आगे बढ़ने के लिए अपनी बैंक पासबुक फोटो भेजें।",
    ],
    "marathi": [
        "तुमचा प्रलंबित कॅशबॅक Rs.{amount} रिलीज करण्यासाठी, कृपया आत्ताच मिळालेला OTP आमच्या एक्झिक्युटिव्हसोबत शेअर करा.",
        "तुमचा रिफंड अडकला आहे, प्रक्रिया पूर्ण करण्यासाठी फोनवर आलेला व्हेरिफिकेशन कोड शेअर करा.",
        "तुमची गॅस सिलेंडर सबसिडी रक्कम तेव्हाच जमा होईल जेव्हा तुम्ही पडताळणीसाठी पाठवलेला OTP शेअर कराल.",
        "तुमची ओळख पडताळायची आहे, कृपया कॉलवर आमच्या सपोर्ट एजंटला OTP सांगा.",
        "तुमची बक्षीस रक्कम तेव्हाच रिलीज होईल जेव्हा तुम्ही मोबाईलवर मिळालेला 6-अंकी कोड शेअर कराल.",
        "{scheme}: लाभार्थी स्थिती पुष्टी करण्यासाठी, नोंदणीकृत नंबरवर आलेला OTP आमच्या हेल्पलाइनला शेअर करा.",
        "तुमचे वीज कनेक्शन आज संध्याकाळी बंद केले जाईल, आमच्या सिस्टममध्ये न भरलेल्या पेमेंटची नोंद आहे.",
        "तुमची पाणी सेवा उद्यापासून बंद केली जाईल, तुमच्या मीटरमध्ये थकीत रक्कम आढळली आहे.",
        "तुमचे DTH रिचार्ज व्हेरिफिकेशन अयशस्वी झाले, 1 तासात पडताळणी न केल्यास कनेक्शन तोडले जाईल.",
        "{scheme}: तुमचा अर्ज होल्डवर आहे, पुढे जाण्यासाठी तुमची बँक पासबुक फोटो पाठवा.",
    ],
}

HAM_TEMPLATES = {
    "english": [
        "Oops, I think I accidentally sent you Rs.{amount} instead of {name}, no rush, send it back whenever you check your phone.",
        "Haha I accidentally sent you Rs.{amount} meant for the group trip fund, ignore or just add it to what you already owe.",
        "Sent you Rs.{amount} by mistake, my bad, keep it and we'll adjust it against dinner next time.",
        "{name}, I think I typed the wrong number and sent Rs.{amount} to you, it's fine, just let me know when you see it.",
        "Your {utility} bill payment of Rs.{amount} was successful this month, thank you.",
        "Your {utility} connection had its scheduled annual inspection today, everything is in order.",
        "Reminder: your {utility} service renewal is due next month, no action needed until then.",
        "{name}, are you around this weekend? Thinking of catching up over coffee.",
        "Can't talk right now, in a meeting, will call you back in an hour.",
        "Don't forget to bring the charger when you come over tonight.",
        "{scheme}: your application status has been updated to 'under review', check the portal for details.",
    ],
    "hinglish": [
        "Oops, mujhe lagta hai maine galti se tumhe Rs.{amount} bhej diye {name} ki jagah, koi jaldi nahi, jab phone check karo tab bhej dena.",
        "Haha maine galti se group trip fund ke liye Rs.{amount} tumhe bhej diye, ignore kar do ya jo already udhaar hai usme adjust kar lena.",
        "Galti se Rs.{amount} bhej diye tumhe, koi baat nahi, rakh lo, agli baar dinner mein adjust kar lenge.",
        "{name}, lagta hai maine galat number type kiya aur Rs.{amount} tumhe bhej diye, koi baat nahi, dekh lo to bata dena.",
        "Aapka {utility} bill payment Rs.{amount} ka is mahine successful raha, dhanyavad.",
        "Aapke {utility} connection ka scheduled annual inspection aaj ho gaya, sab kuch theek hai.",
        "Reminder: aapki {utility} service renewal agle mahine due hai, abhi koi action nahi chahiye.",
        "{name}, is weekend free ho? Coffee pe milte hain sochta hoon.",
        "Abhi baat nahi kar sakta, meeting mein hoon, ek ghante mein call karta hoon.",
        "Aaj raat aate waqt charger lana mat bhoolna.",
        "{scheme}: aapke application ka status 'under review' update hua hai, details ke liye portal check karein.",
    ],
    "hindi": [
        "उफ्फ, मुझे लगता है मैंने गलती से {name} की जगह तुम्हें Rs.{amount} भेज दिए, कोई जल्दी नहीं, जब फोन चेक करो तब भेज देना।",
        "हाहा मैंने गलती से ग्रुप ट्रिप फंड के लिए Rs.{amount} तुम्हें भेज दिए, इग्नोर कर दो या जो पहले से उधार है उसमें एडजस्ट कर लेना।",
        "गलती से Rs.{amount} भेज दिए तुम्हें, कोई बात नहीं, रख लो, अगली बार डिनर में एडजस्ट कर लेंगे।",
        "{name}, लगता है मैंने गलत नंबर टाइप किया और Rs.{amount} तुम्हें भेज दिए, कोई बात नहीं, देख लो तो बता देना।",
        "आपका {utility} बिल भुगतान Rs.{amount} का इस महीने सफल रहा, धन्यवाद।",
        "आपके {utility} कनेक्शन का निर्धारित वार्षिक निरीक्षण आज हो गया, सब कुछ ठीक है।",
        "अनुस्मारक: आपकी {utility} सेवा नवीनीकरण अगले महीने देय है, अभी कोई कार्रवाई आवश्यक नहीं है।",
        "{name}, इस वीकेंड फ्री हो? कॉफी पर मिलते हैं सोच रहा था।",
        "अभी बात नहीं कर सकता, मीटिंग में हूं, एक घंटे में कॉल करता हूं।",
        "आज रात आते समय चार्जर लाना मत भूलना।",
        "{scheme}: आपके आवेदन की स्थिति 'समीक्षाधीन' अपडेट हुई है, विवरण के लिए पोर्टल देखें।",
    ],
    "marathi": [
        "अरेरे, मला वाटते मी चुकून {name} ऐवजी तुला Rs.{amount} पाठवले, घाई नाही, फोन चेक केल्यावर पाठव.",
        "हाहा मी चुकून ग्रुप ट्रिप फंडासाठी Rs.{amount} तुला पाठवले, दुर्लक्ष कर किंवा आधीच्या उधारीत अ‍ॅडजस्ट कर.",
        "चुकून Rs.{amount} पाठवले तुला, हरकत नाही, ठेव, पुढच्या वेळी डिनरमध्ये अ‍ॅडजस्ट करू.",
        "{name}, वाटते मी चुकीचा नंबर टाईप केला आणि Rs.{amount} तुला पाठवले, हरकत नाही, दिसल्यावर सांग.",
        "तुमचे {utility} बिल पेमेंट Rs.{amount} चे या महिन्यात यशस्वी झाले, धन्यवाद.",
        "तुमच्या {utility} कनेक्शनची नियोजित वार्षिक तपासणी आज झाली, सर्व काही व्यवस्थित आहे.",
        "आठवण: तुमचे {utility} सेवा नूतनीकरण पुढच्या महिन्यात देय आहे, आत्ता काही करण्याची गरज नाही.",
        "{name}, या वीकेंडला फ्री आहेस का? कॉफीसाठी भेटूया विचार करत होतो.",
        "आत्ता बोलू शकत नाही, मीटिंगमध्ये आहे, तासाभरात कॉल करतो.",
        "आज रात्री येताना चार्जर आणायला विसरू नकोस.",
        "{scheme}: तुमच्या अर्जाची स्थिती 'पुनरावलोकनाधीन' अपडेट झाली आहे, तपशीलासाठी पोर्टल पहा.",
    ],
}

PER_LANG_TARGET = 4000

if __name__ == "__main__":
    summary = []
    for lang in ["english", "hinglish", "hindi", "marathi"]:
        for label, templates, fname in [
            ("spam", SPAM_TEMPLATES[lang], f"Synthetic_Fraud2_Spam_{lang.capitalize()}.csv"),
            ("ham", HAM_TEMPLATES[lang], f"Synthetic_Fraud2_Ham_{lang.capitalize()}.csv"),
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

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "augment_fraud_patterns_v2_result.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
