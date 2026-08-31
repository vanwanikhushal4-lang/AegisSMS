# -*- coding: utf-8 -*-
"""
Fifth-iteration data augmentation, added after a user report that genuine
promotional messages from real businesses (sale announcements, loyalty
rewards, subscription renewal upsells, cart reminders) were being
misclassified as Spam. The training data's only notion of "promotional"
text was spam (generic "Dear Customer" broadcasts, fake prize draws, calls
to an unknown phone number, no real brand/domain) -- there was no
legitimate-promotional Ham category at all, so the model learned
"discount/sale/offer language => spam" too broadly.

This batch adds diverse legitimate promotional Ham messages: real,
well-known brand names paired with a plausible domain of that brand
(reusing the "no hardcoded trust whitelist, let the model learn from
co-occurrence" approach from augment_ham_url.py), across many industries
(e-commerce, food delivery, OTT/streaming, telecom, banking/fintech,
travel, fashion/beauty, edtech, fitness, insurance, pharmacy,
entertainment, furniture) and writing styles (flash sale, cart reminder,
loyalty points, subscription renewal upsell, festival greeting, birthday
voucher, referral, app-only deal, formal with opt-out). Urgency/scarcity
phrasing ("today only", "hurry", "limited period") is deliberately
included in some legit examples too, since real brands use it just as
often as scammers -- the point is for the model to key off brand +
legitimate domain + absence of prize-claim/credential-request framing,
not off urgency words alone.

A smaller matched contrastive Spam set (brand-impersonation promotional
scams: fake lucky draws, "call this number to claim", suspicious
lookalike links) is included so the model doesn't overcorrect into
trusting any message that merely mentions a brand name.
"""
import csv
import os
import random
import string

random.seed(47)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset_5971")

NAMES = {
    "english": ["Raj", "Priya", "Amit", "Sneha", "Rahul", "Anita", "Vikram", "Pooja"],
    "hinglish": ["Raj", "Priya", "Amit", "Sneha", "Rahul", "Anita", "Vikram", "Pooja"],
    "hindi": ["राज", "प्रिया", "अमित", "स्नेहा", "राहुल", "अनीता", "विक्रम", "पूजा"],
    "marathi": ["राज", "प्रिया", "अमोल", "स्नेहल", "राहुल", "अनिता", "विक्रम", "पूजा"],
}

FESTIVALS = {
    "english": ["Diwali", "Holi", "New Year", "Eid", "Christmas", "Independence Day"],
    "hinglish": ["Diwali", "Holi", "New Year", "Eid", "Christmas"],
    "hindi": ["दिवाली", "होली", "नए साल", "ईद", "क्रिसमस"],
    "marathi": ["दिवाळी", "होळी", "नवीन वर्ष", "ईद", "ख्रिसमस"],
}

BRAND_GROUPS = {
    "retail": [
        ("Amazon", "amazon.in/deals"), ("Flipkart", "flipkart.com/sale"),
        ("Myntra", "myntra.com/offers"), ("Ajio", "ajio.com/sale"),
        ("Meesho", "meesho.com/offers"), ("Nykaa", "nykaa.com/sale"),
        ("BigBasket", "bigbasket.com/offers"), ("Zepto", "zepto.com/offers"),
        ("Blinkit", "blinkit.com/offers"), ("Swiggy", "swiggy.com/offers"),
        ("Zomato", "zomato.com/gold"), ("Domino's", "dominos.co.in/offers"),
        ("Titan", "titan.co.in/sale"), ("Tanishq", "tanishq.co.in/offers"),
        ("Decathlon", "decathlon.in/sale"), ("boAt", "boat-lifestyle.com/sale"),
        ("Lenskart", "lenskart.com/offers"), ("PharmEasy", "pharmeasy.in/offers"),
        ("1mg", "1mg.com/offers"), ("Pepperfry", "pepperfry.com/sale"),
        ("BookMyShow", "bookmyshow.com/offers"),
    ],
    "subscription": [
        ("Netflix", "netflix.com/account"), ("Amazon Prime Video", "primevideo.com/offers"),
        ("Disney+ Hotstar", "hotstar.com/subscribe"), ("Spotify", "spotify.com/premium"),
        ("Airtel", "airtel.in/offers"), ("Jio", "jio.com/offers"), ("Vi", "myvi.in/offers"),
        ("cult.fit", "cult.fit/offers"), ("BYJU'S", "byjus.com/offers"),
        ("Unacademy", "unacademy.com/offers"), ("Policybazaar", "policybazaar.com/renew"),
    ],
    "fintech": [
        ("HDFC Bank", "hdfcbank.com/offers"), ("ICICI Bank", "icicibank.com/offers"),
        ("SBI Card", "sbicard.com/offers"), ("Axis Bank", "axisbank.com/offers"),
        ("Paytm", "paytm.com/rewards"), ("PhonePe", "phonepe.com/rewards"),
        ("CRED", "cred.club/rewards"),
    ],
    "travel": [
        ("IRCTC", "irctc.co.in/offers"), ("MakeMyTrip", "makemytrip.com/deals"),
        ("Goibibo", "goibibo.com/deals"), ("IndiGo", "goindigo.in/offers"),
        ("Ola", "olacabs.com/offers"), ("Uber", "uber.com/promotions"),
        ("redBus", "redbus.in/offers"),
    ],
}

SUSPICIOUS_DOMAINS = ["reward-claim24.xyz", "prize-verify.info", "lucky-winner-alert.co",
                      "offer4u-claim.online", "bonus-cashout.xyz", "gift-redeem-now.co.in"]

CATEGORIES = {
    "english": ["electronics", "fashion", "footwear", "beauty products", "home decor",
                "accessories", "groceries", "kitchen appliances"],
    "hinglish": ["electronics", "fashion", "footwear", "beauty products", "home decor",
                 "accessories", "groceries", "kitchen appliances"],
    "hindi": ["इलेक्ट्रॉनिक्स", "फैशन", "फुटवियर", "ब्यूटी प्रोडक्ट्स", "होम डेकोर",
              "एक्सेसरीज", "किराना", "किचन अप्लायंसेज"],
    "marathi": ["इलेक्ट्रॉनिक्स", "फॅशन", "फुटवेअर", "ब्यूटी प्रोडक्ट्स", "होम डेकोर",
                "अ‍ॅक्सेसरीज", "किराणा", "किचन अप्लायन्सेस"],
}

PRIZES = {
    "english": ["smartphone", "smartwatch", "gift hamper", "gold coin", "shopping voucher"],
    "hinglish": ["smartphone", "smartwatch", "gift hamper", "sona coin", "shopping voucher"],
    "hindi": ["स्मार्टफोन", "स्मार्टवॉच", "गिफ्ट हैम्पर", "सोने का सिक्का", "शॉपिंग वाउचर"],
    "marathi": ["स्मार्टफोन", "स्मार्टवॉच", "गिफ्ट हॅम्पर", "सोन्याचे नाणे", "शॉपिंग व्हाउचर"],
}

SALE_EVENTS = {
    "english": ["Big Billion Days", "Great Indian Festival", "End of Season Sale",
                "Diwali Dhamaka Sale", "Republic Day Sale", "Independence Day Sale",
                "Year End Sale", "Monsoon Sale"],
    "hinglish": ["Big Billion Days", "Great Indian Festival", "End of Season Sale",
                 "Diwali Dhamaka Sale", "Republic Day Sale", "Independence Day Sale",
                 "Year End Sale", "Monsoon Sale"],
    "hindi": ["बिग बिलियन डेज़", "ग्रेट इंडियन फेस्टिवल", "एंड ऑफ सीजन सेल",
              "दिवाली धमाका सेल", "रिपब्लिक डे सेल", "इंडिपेंडेंस डे सेल",
              "ईयर एंड सेल", "मानसून सेल"],
    "marathi": ["बिग बिलियन डेज", "ग्रेट इंडियन फेस्टिव्हल", "एंड ऑफ सीझन सेल",
                "दिवाळी धमाका सेल", "रिपब्लिक डे सेल", "इंडिपेंडन्स डे सेल",
                "इयर एंड सेल", "मान्सून सेल"],
}


def rand_pct():
    return str(random.choice([10, 15, 20, 25, 30, 40, 50, 60, 70]))


def rand_mega_pct():
    return str(random.choice([50, 60, 70, 75, 80]))


def rand_amount():
    return f"{random.choice([50, 100, 150, 200, 250, 300, 500]):,}"


def rand_points():
    return str(random.choice([100, 150, 200, 250, 300, 500, 750, 1000]))


def rand_date():
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.choice([2025, 2026])
    return f"{day:02d}-{month:02d}-{year}"


def rand_code():
    return "".join(random.choices(string.ascii_uppercase, k=4)) + str(random.randint(10, 99))


def rand_legit_url(domain):
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    return f"https://{domain}/{suffix}"


def rand_suspicious_url():
    domain = random.choice(SUSPICIOUS_DOMAINS)
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return f"http://{domain}/{suffix}"


def rand_phone():
    first = random.choice("6789")
    rest = ''.join(random.choices(string.digits, k=9))
    return f"{first}{rest}"


def fill(template, lang, group):
    brand, domain = random.choice(BRAND_GROUPS[group])
    return template.format(
        brand=brand, legit_url=rand_legit_url(domain), pct=rand_pct(), amount=rand_amount(),
        points=rand_points(), date=rand_date(), code=rand_code(), name=random.choice(NAMES[lang]),
        category=random.choice(CATEGORIES[lang]), festival=random.choice(FESTIVALS[lang]),
        mega_pct=rand_mega_pct(), sale_event=random.choice(SALE_EVENTS[lang]),
    ), brand


def fill_spam(template, lang):
    return template.format(
        brand=random.choice([b for grp in BRAND_GROUPS.values() for b, _ in grp]),
        amount=rand_amount(), phone=rand_phone(), url=rand_suspicious_url(),
        prize=random.choice(PRIZES[lang]), code=rand_code(),
    )


# ---------------------------------------------------------------------------
# HAM: legitimate promotional templates, grouped by which brand pool they fit
# ---------------------------------------------------------------------------
HAM_TEMPLATES = {
    "english": {
        "retail": [
            "{brand}'s Big Sale is live! Up to {pct}% off on {category}. Shop now: {legit_url}",
            "Hi {name}, the items in your {brand} cart are now on sale. Complete your order: {legit_url}",
            "New arrivals just dropped on {brand}! Check out the latest {category} collection: {legit_url}",
            "Loved your last {brand} order? Here's {pct}% off your next one, valid till {date}: {legit_url}",
            "{brand}: Flat {pct}% off storewide this weekend. T&C apply. Reply STOP to unsubscribe from offers. {legit_url}",
            "Open the {brand} app now for an exclusive app-only deal of {pct}% off, today only: {legit_url}",
            "Happy Birthday {name}! Here's a Rs.{amount} {brand} voucher just for you, valid this month: {legit_url}",
            "Wishing you a Happy {festival}! Celebrate with {pct}% off at {brand}, only for our valued customers: {legit_url}",
            "Invite a friend to {brand} and you both get Rs.{amount} off your next order. Share code {code}: {legit_url}",
            "{brand}: Your wishlist items are now {pct}% off, hurry before they sell out: {legit_url}",
            "{brand} {sale_event} is LIVE! Up to {mega_pct}% off on {category}, hurry before stock runs out: {legit_url}",
            "Don't miss {brand}'s {sale_event} - up to {mega_pct}% off storewide, only for a limited time: {legit_url}",
            "{sale_event} at {brand}: flat up to {mega_pct}% off on {category}, download the app now: {legit_url}",
            "Last few hours of {brand}'s {sale_event}! Up to {mega_pct}% off, shop before it ends tonight: {legit_url}",
            "Only for today: {brand} is offering flat {pct}% off on {category}, order before midnight: {legit_url}",
        ],
        "subscription": [
            "Your {brand} subscription renews on {date}. Upgrade to the annual plan and save {pct}%: {legit_url}",
            "As a {brand} member, enjoy early access to our {pct}% off renewal offer starting today: {legit_url}",
            "{brand}: Add a second screen to your plan for just Rs.{amount} more per month. Manage your plan: {legit_url}",
            "Your {brand} plan is expiring soon. Renew now and get {pct}% off your first 3 months: {legit_url}",
            "Limited period: upgrade your {brand} membership today and save Rs.{amount} on the annual fee: {legit_url}",
            "Last chance: your {brand} bonus data offer expires tonight. Recharge now to keep the extra benefit: {legit_url}",
            "Your {brand} plan benefit ends today. Renew before midnight to avoid losing your extra data: {legit_url}",
        ],
        "fintech": [
            "You've earned {points} {brand} reward points. Redeem them on your next transaction: {legit_url}",
            "{brand}: Get {pct}% cashback on bill payments made through the app this week. Details: {legit_url}",
            "Your {brand} card is eligible for a fee waiver this year on spending Rs.{amount}+. Check details: {legit_url}",
            "{brand}: Refer a friend and both of you get {points} reward points. Share your code {code}: {legit_url}",
        ],
        "travel": [
            "{brand}: Book now and get {pct}% off your next trip, valid till {date}. Details: {legit_url}",
            "Flash sale on {brand}! Fares starting at Rs.{amount} for a limited period. Book here: {legit_url}",
            "{brand} members get early access to our festive sale, {pct}% off select routes: {legit_url}",
            "Plan your next trip with {brand} and save Rs.{amount} using code {code}: {legit_url}",
        ],
    },
    "hinglish": {
        "retail": [
            "{brand} ki Big Sale live hai! {category} par {pct}% tak ki chhoot. Abhi shop karein: {legit_url}",
            "Hi {name}, aapke {brand} cart ke items ab sale mein hain. Order complete karein: {legit_url}",
            "{brand} par naye arrivals aa gaye hain! Latest {category} collection dekhein: {legit_url}",
            "Aapka pichla {brand} order pasand aaya tha? Agli order par {pct}% off, {date} tak valid: {legit_url}",
            "{brand}: Is weekend storewide flat {pct}% off. T&C apply. Offers band karne ke liye STOP reply karein. {legit_url}",
            "{brand} app abhi kholein aur pao exclusive app-only deal {pct}% off, sirf aaj: {legit_url}",
            "Happy Birthday {name}! Aapke liye Rs.{amount} ka {brand} voucher, is mahine valid: {legit_url}",
            "{festival} ki hardik shubhkamnayein! {brand} par {pct}% off ke saath celebrate karein, sirf hamare valued customers ke liye: {legit_url}",
            "{brand} par dost ko invite karein aur dono ko Rs.{amount} off milega. Code {code} share karein: {legit_url}",
            "{brand} {sale_event} LIVE hai! {category} par {mega_pct}% tak ki chhoot, stock khatam hone se pehle jaldi karein: {legit_url}",
            "{brand} ki {sale_event} miss mat karein - storewide {mega_pct}% tak off, sirf limited time ke liye: {legit_url}",
            "{sale_event} sirf {brand} par: {category} par flat {mega_pct}% tak chhoot, abhi app download karein: {legit_url}",
            "{brand} ki {sale_event} ke aakhri kuch ghante! {mega_pct}% tak off, aaj raat khatam hone se pehle shop karein: {legit_url}",
            "Sirf aaj ke liye: {brand} par {category} par flat {pct}% off mil raha hai, aadhi raat se pehle order karein: {legit_url}",
        ],
        "subscription": [
            "Aapka {brand} subscription {date} ko renew hoga. Annual plan par upgrade karein aur {pct}% bachayein: {legit_url}",
            "{brand} member hone ke naate, aaj se hamare {pct}% off renewal offer ka early access paayein: {legit_url}",
            "Aapka {brand} plan jald expire ho raha hai. Abhi renew karein aur pehle 3 mahino par {pct}% off paayein: {legit_url}",
            "Limited period: aaj hi apni {brand} membership upgrade karein aur annual fee par Rs.{amount} bachayein: {legit_url}",
            "Aakhri mauka: aapke {brand} bonus data offer aaj raat expire ho raha hai. Extra benefit ke liye abhi recharge karein: {legit_url}",
            "Aapka {brand} plan benefit aaj khatam ho raha hai. Extra data na khone ke liye midnight se pehle renew karein: {legit_url}",
        ],
        "fintech": [
            "Aapne {points} {brand} reward points kamaye hain. Agli transaction par redeem karein: {legit_url}",
            "{brand}: Is hafte app se bill payment karne par {pct}% cashback paayein. Details: {legit_url}",
            "{brand}: Dost ko refer karein aur dono ko {points} reward points milenge. Code {code} share karein: {legit_url}",
        ],
        "travel": [
            "{brand}: Abhi book karein aur agli trip par {pct}% off paayein, {date} tak valid. Details: {legit_url}",
            "{brand} par flash sale! Fares Rs.{amount} se shuru, limited period ke liye. Yahan book karein: {legit_url}",
            "{brand} members ko hamare festive sale ka early access milta hai, select routes par {pct}% off: {legit_url}",
        ],
    },
    "hindi": {
        "retail": [
            "{brand} की बिग सेल लाइव है! {category} पर {pct}% तक की छूट। अभी शॉप करें: {legit_url}",
            "हाय {name}, आपके {brand} कार्ट के आइटम अब सेल में हैं। ऑर्डर पूरा करें: {legit_url}",
            "{brand} पर नए अराइवल्स आ गए हैं! नवीनतम {category} कलेक्शन देखें: {legit_url}",
            "आपका पिछला {brand} ऑर्डर पसंद आया था? अगले ऑर्डर पर {pct}% छूट, {date} तक वैध: {legit_url}",
            "{brand}: इस वीकेंड स्टोरवाइड फ्लैट {pct}% छूट। नियम व शर्तें लागू। ऑफर बंद करने के लिए STOP भेजें। {legit_url}",
            "{brand} ऐप अभी खोलें और पाएं एक्सक्लूसिव ऐप-ओनली डील {pct}% छूट, सिर्फ आज: {legit_url}",
            "जन्मदिन मुबारक {name}! आपके लिए Rs.{amount} का {brand} वाउचर, इस महीने वैध: {legit_url}",
            "{festival} की हार्दिक शुभकामनाएं! {brand} पर {pct}% छूट के साथ मनाएं, सिर्फ हमारे वैल्यूड कस्टमर्स के लिए: {legit_url}",
            "{brand} पर दोस्त को इनवाइट करें और दोनों को Rs.{amount} की छूट मिलेगी। कोड {code} शेयर करें: {legit_url}",
            "{brand} {sale_event} लाइव है! {category} पर {mega_pct}% तक की छूट, स्टॉक खत्म होने से पहले जल्दी करें: {legit_url}",
            "{brand} की {sale_event} मिस न करें - स्टोरवाइड {mega_pct}% तक की छूट, सिर्फ सीमित समय के लिए: {legit_url}",
            "{sale_event} सिर्फ {brand} पर: {category} पर फ्लैट {mega_pct}% तक छूट, अभी ऐप डाउनलोड करें: {legit_url}",
            "{brand} की {sale_event} के आखिरी कुछ घंटे! {mega_pct}% तक की छूट, आज रात खत्म होने से पहले शॉप करें: {legit_url}",
            "सिर्फ आज के लिए: {brand} पर {category} पर फ्लैट {pct}% छूट मिल रही है, आधी रात से पहले ऑर्डर करें: {legit_url}",
        ],
        "subscription": [
            "आपका {brand} सब्सक्रिप्शन {date} को रिन्यू होगा। एनुअल प्लान पर अपग्रेड करें और {pct}% बचाएं: {legit_url}",
            "{brand} सदस्य होने के नाते, आज से हमारे {pct}% छूट रिन्यूअल ऑफर का जल्दी एक्सेस पाएं: {legit_url}",
            "आपका {brand} प्लान जल्द समाप्त हो रहा है। अभी रिन्यू करें और पहले 3 महीनों पर {pct}% छूट पाएं: {legit_url}",
            "सीमित समय: आज ही अपनी {brand} मेंबरशिप अपग्रेड करें और एनुअल फीस पर Rs.{amount} बचाएं: {legit_url}",
            "आखिरी मौका: आपके {brand} बोनस डेटा ऑफर की वैधता आज रात समाप्त हो रही है। एक्स्ट्रा बेनिफिट के लिए अभी रिचार्ज करें: {legit_url}",
            "आपके {brand} प्लान का बेनिफिट आज खत्म हो रहा है। एक्स्ट्रा डेटा न खोने के लिए मध्यरात्रि से पहले रिन्यू करें: {legit_url}",
        ],
        "fintech": [
            "आपने {points} {brand} रिवॉर्ड पॉइंट्स कमाए हैं। अगले ट्रांजैक्शन पर रिडीम करें: {legit_url}",
            "{brand}: इस हफ्ते ऐप से बिल पेमेंट करने पर {pct}% कैशबैक पाएं। विवरण: {legit_url}",
            "{brand}: दोस्त को रेफर करें और दोनों को {points} रिवॉर्ड पॉइंट्स मिलेंगे। कोड {code} शेयर करें: {legit_url}",
        ],
        "travel": [
            "{brand}: अभी बुक करें और अगली ट्रिप पर {pct}% छूट पाएं, {date} तक वैध। विवरण: {legit_url}",
            "{brand} पर फ्लैश सेल! किराया Rs.{amount} से शुरू, सीमित समय के लिए। यहां बुक करें: {legit_url}",
            "{brand} सदस्यों को हमारे फेस्टिव सेल का जल्दी एक्सेस मिलता है, चुनिंदा रूट्स पर {pct}% छूट: {legit_url}",
        ],
    },
    "marathi": {
        "retail": [
            "{brand} ची बिग सेल लाइव्ह आहे! {category} वर {pct}% पर्यंत सूट. आत्ताच शॉप करा: {legit_url}",
            "हाय {name}, तुमच्या {brand} कार्टमधील वस्तू आता सेलमध्ये आहेत. ऑर्डर पूर्ण करा: {legit_url}",
            "{brand} वर नवीन अरायव्हल्स आले आहेत! नवीनतम {category} कलेक्शन पहा: {legit_url}",
            "तुमची मागची {brand} ऑर्डर आवडली होती? पुढच्या ऑर्डरवर {pct}% सूट, {date} पर्यंत वैध: {legit_url}",
            "{brand}: या वीकेंडला स्टोअरवाइड फ्लॅट {pct}% सूट. अटी लागू. ऑफर्स बंद करण्यासाठी STOP पाठवा. {legit_url}",
            "{brand} अ‍ॅप आत्ताच उघडा आणि मिळवा एक्सक्लुझिव्ह अ‍ॅप-ओन्ली डील {pct}% सूट, फक्त आज: {legit_url}",
            "वाढदिवसाच्या शुभेच्छा {name}! तुमच्यासाठी Rs.{amount} चा {brand} व्हाउचर, या महिन्यात वैध: {legit_url}",
            "{festival} च्या हार्दिक शुभेच्छा! {brand} वर {pct}% सूट सह साजरा करा, फक्त आमच्या व्हॅल्यूड कस्टमर्ससाठी: {legit_url}",
            "{brand} वर मित्राला आमंत्रित करा आणि दोघांनाही Rs.{amount} सूट मिळेल. कोड {code} शेअर करा: {legit_url}",
            "{brand} {sale_event} लाइव्ह आहे! {category} वर {mega_pct}% पर्यंत सूट, स्टॉक संपण्यापूर्वी घाई करा: {legit_url}",
            "{brand} ची {sale_event} चुकवू नका - स्टोअरवाइड {mega_pct}% पर्यंत सूट, फक्त मर्यादित काळासाठी: {legit_url}",
            "{sale_event} फक्त {brand} वर: {category} वर फ्लॅट {mega_pct}% पर्यंत सूट, आत्ताच अ‍ॅप डाउनलोड करा: {legit_url}",
            "{brand} च्या {sale_event} चे शेवटचे काही तास! {mega_pct}% पर्यंत सूट, आज रात्री संपण्यापूर्वी शॉप करा: {legit_url}",
            "फक्त आजसाठी: {brand} वर {category} वर फ्लॅट {pct}% सूट मिळत आहे, मध्यरात्रीपूर्वी ऑर्डर करा: {legit_url}",
        ],
        "subscription": [
            "तुमचे {brand} सबस्क्रिप्शन {date} रोजी रिन्यू होईल. वार्षिक प्लॅनवर अपग्रेड करा आणि {pct}% वाचवा: {legit_url}",
            "{brand} सदस्य असल्याने, आजपासून आमच्या {pct}% सूट रिन्यूवल ऑफरचा लवकर अ‍ॅक्सेस मिळवा: {legit_url}",
            "तुमचा {brand} प्लॅन लवकरच संपत आहे. आत्ताच रिन्यू करा आणि पहिल्या 3 महिन्यांवर {pct}% सूट मिळवा: {legit_url}",
            "मर्यादित काळ: आजच तुमची {brand} मेंबरशिप अपग्रेड करा आणि वार्षिक फीवर Rs.{amount} वाचवा: {legit_url}",
            "शेवटची संधी: तुमच्या {brand} बोनस डेटा ऑफरची वैधता आज रात्री संपत आहे. एक्स्ट्रा बेनिफिटसाठी आत्ताच रिचार्ज करा: {legit_url}",
            "तुमच्या {brand} प्लॅनचा बेनिफिट आज संपत आहे. एक्स्ट्रा डेटा गमावू नये म्हणून मध्यरात्रीपूर्वी रिन्यू करा: {legit_url}",
        ],
        "fintech": [
            "तुम्ही {points} {brand} रिवॉर्ड पॉइंट्स कमावले आहेत. पुढच्या व्यवहारावर रिडीम करा: {legit_url}",
            "{brand}: या आठवड्यात अ‍ॅपमधून बिल पेमेंट केल्यास {pct}% कॅशबॅक मिळवा. तपशील: {legit_url}",
            "{brand}: मित्राला रेफर करा आणि दोघांनाही {points} रिवॉर्ड पॉइंट्स मिळतील. कोड {code} शेअर करा: {legit_url}",
        ],
        "travel": [
            "{brand}: आत्ताच बुक करा आणि पुढच्या ट्रिपवर {pct}% सूट मिळवा, {date} पर्यंत वैध. तपशील: {legit_url}",
            "{brand} वर फ्लॅश सेल! भाडे Rs.{amount} पासून सुरू, मर्यादित काळासाठी. इथे बुक करा: {legit_url}",
            "{brand} सदस्यांना आमच्या फेस्टिव्ह सेलचा लवकर अ‍ॅक्सेस मिळतो, निवडक मार्गांवर {pct}% सूट: {legit_url}",
        ],
    },
}

# ---------------------------------------------------------------------------
# SPAM (contrastive): brand-impersonation promotional scams -- mentions a
# real brand name but shows the classic tells (unknown call-in phone number,
# fake prize/lucky draw, suspicious lookalike link, no real domain)
# ---------------------------------------------------------------------------
SPAM_TEMPLATES = {
    "english": [
        "{brand} Lucky Draw: You've won a free {prize}! Call {phone} now to claim before it expires.",
        "Congratulations! Your {brand} account is selected for a free {prize} worth Rs.{amount}. Claim now: {url}",
        "{brand} Flash Sale: 90% off everything, only a few pieces left! Reply BUY to order now, limited stock.",
        "You are the lucky winner of {brand}'s anniversary contest. Share your bank details to receive your Rs.{amount} prize.",
        "{brand} customer service: your refund of Rs.{amount} is pending, click to claim: {url}",
        "{brand} is giving away free {prize} to 100 lucky customers today. Call {phone} immediately to grab yours.",
    ],
    "hinglish": [
        "{brand} Lucky Draw: Aapne ek free {prize} jeeta hai! Expire hone se pehle abhi call karein {phone}.",
        "Badhai ho! Aapka {brand} account ek free {prize} (Rs.{amount}) ke liye select hua hai. Abhi claim karein: {url}",
        "{brand} Flash Sale: sab kuch 90% off, sirf kuch pieces bache hain! Abhi order karne ke liye BUY reply karein.",
        "Aap {brand} ke anniversary contest ke lucky winner hain. Rs.{amount} ka prize paane ke liye apni bank details share karein.",
        "{brand} customer service: aapka Rs.{amount} ka refund pending hai, claim karne ke liye click karein: {url}",
        "{brand} aaj 100 lucky customers ko free {prize} de raha hai. Apna paane ke liye turant call karein {phone}.",
    ],
    "hindi": [
        "{brand} लकी ड्रॉ: आपने एक फ्री {prize} जीता है! एक्सपायर होने से पहले अभी कॉल करें {phone}।",
        "बधाई हो! आपका {brand} खाता एक फ्री {prize} (Rs.{amount}) के लिए चुना गया है। अभी दावा करें: {url}",
        "{brand} फ्लैश सेल: सब कुछ 90% छूट, सिर्फ कुछ पीस बचे हैं! अभी ऑर्डर करने के लिए BUY रिप्लाई करें।",
        "आप {brand} के एनिवर्सरी कॉन्टेस्ट के लकी विनर हैं। Rs.{amount} का प्राइज पाने के लिए अपनी बैंक विवरण साझा करें।",
        "{brand} कस्टमर सर्विस: आपका Rs.{amount} का रिफंड पेंडिंग है, दावा करने के लिए क्लिक करें: {url}",
        "{brand} आज 100 लकी कस्टमर्स को फ्री {prize} दे रहा है। अपना पाने के लिए तुरंत कॉल करें {phone}।",
    ],
    "marathi": [
        "{brand} लकी ड्रॉ: तुम्ही एक फ्री {prize} जिंकला आहे! एक्सपायर होण्यापूर्वी आत्ताच कॉल करा {phone}.",
        "अभिनंदन! तुमचे {brand} खाते एका फ्री {prize} (Rs.{amount}) साठी निवडले गेले आहे. आत्ताच दावा करा: {url}",
        "{brand} फ्लॅश सेल: सर्व काही 90% सूट, फक्त काही पीसेस शिल्लक! आत्ताच ऑर्डर करण्यासाठी BUY रिप्लाय करा.",
        "तुम्ही {brand} च्या अ‍ॅनिव्हर्सरी कॉन्टेस्टचे लकी विनर आहात. Rs.{amount} चे बक्षीस मिळवण्यासाठी तुमची बँक माहिती शेअर करा.",
        "{brand} कस्टमर सर्व्हिस: तुमचा Rs.{amount} चा रिफंड प्रलंबित आहे, दावा करण्यासाठी क्लिक करा: {url}",
        "{brand} आज 100 लकी कस्टमर्सना फ्री {prize} देत आहे. तुमचे मिळवण्यासाठी लगेच कॉल करा {phone}.",
    ],
}

PER_LANG_TARGET_HAM = 5000
PER_LANG_TARGET_SPAM = 2000

if __name__ == "__main__":
    summary = []
    for lang in ["english", "hinglish", "hindi", "marathi"]:
        # ---- legitimate promotional ham ----
        rows = []
        seen = set()
        attempts = 0
        max_attempts = PER_LANG_TARGET_HAM * 40
        groups = list(HAM_TEMPLATES[lang].keys())
        while len(rows) < PER_LANG_TARGET_HAM and attempts < max_attempts:
            attempts += 1
            group = random.choice(groups)
            template = random.choice(HAM_TEMPLATES[lang][group])
            text, brand = fill(template, lang, group)
            if text in seen:
                continue
            seen.add(text)
            rows.append({"LABEL": "ham", "TEXT": text, "URL": "yes", "EMAIL": "No", "PHONE": "No"})

        out_path = os.path.join(OUT_DIR, f"Synthetic_PromoLegit_Ham_{lang.capitalize()}.csv")
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["LABEL", "TEXT", "URL", "EMAIL", "PHONE"])
            writer.writeheader()
            writer.writerows(rows)
        summary.append(f"{lang} promo_ham: {len(rows)} rows -> {out_path}")

        # ---- brand-impersonation promotional spam (contrast) ----
        rows = []
        seen = set()
        attempts = 0
        max_attempts = PER_LANG_TARGET_SPAM * 40
        templates = SPAM_TEMPLATES[lang]
        while len(rows) < PER_LANG_TARGET_SPAM and attempts < max_attempts:
            attempts += 1
            template = random.choice(templates)
            text = fill_spam(template, lang)
            if text in seen:
                continue
            seen.add(text)
            has_url = "{url}" in template
            has_phone = "{phone}" in template
            rows.append({"LABEL": "spam", "TEXT": text, "URL": "yes" if has_url else "No",
                         "EMAIL": "No", "PHONE": "yes" if has_phone else "No"})

        out_path = os.path.join(OUT_DIR, f"Synthetic_PromoLegit_SpamContrast_{lang.capitalize()}.csv")
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["LABEL", "TEXT", "URL", "EMAIL", "PHONE"])
            writer.writeheader()
            writer.writerows(rows)
        summary.append(f"{lang} promo_spam_contrast: {len(rows)} rows -> {out_path}")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "augment_promo_legit_result.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
