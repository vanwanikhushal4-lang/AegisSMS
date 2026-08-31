package com.payshield.aegissms;

import java.text.Normalizer;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class AegisSmsClassifier {

    private static final Pattern VPA_RE = Pattern.compile("\\b[a-zA-Z0-9.\\-_]+@[a-zA-Z0-9.\\-_]+\\b");
    private static final Pattern URL_RE = Pattern.compile(
        "(https?://\\S+|www\\.\\S+|(?:[a-zA-Z0-9-]+\\.)+(?:com|in|org|net|co|gov|edu|io|ai|xyz|top|site|online|apk|app|live|me|ly|link|info)(?:/\\S*)?)",
        Pattern.CASE_INSENSITIVE
    );
    private static final Pattern PHONE_RE = Pattern.compile("(\\+?\\d[\\d\\- ]{7,}\\d)");
    private static final Pattern ZERO_WIDTH_RE = Pattern.compile("[\\u200B-\\u200D\\uFEFF\\u200E\\u200F\\u00AD]");
    private static final Pattern TOKEN_RE = Pattern.compile("[\\p{L}\\p{N}_]+");

    private static final String[] URGENCY_KEYWORDS = {
        "urgent", "immediately", "block", "suspend", "verify now", "verify immediately",
        "kyc", "winner", "lucky draw", "claim now", "act now", "penalty",
        "congratulations you", "confirm now", "update now", "turant verify",
        "will be blocked", "will be suspended", "will be disconnected",
        "will be terminated", "will be discontinued", "will be cut off",
        "connection will be", "pending verification", "meter verification",
        "service will be", "ready for transfer", "final notice", "account locked",
        "pay immediately", "disconnect tonight", "call immediately",
        "pending challan", "challan", "legal action", "court notice", "traffic fine",
        "parivahan", "avoid legal action", "avoid legal disputes", "penalty",
        "तुरंत", "ब्लॉक हो जाएगा", "सत्यापित करें", "जीत चुका", "केवाईसी",
        "निलंबित", "अभी वेरीफाई", "अभिनंदन! आप", "लगेच केवायसी",
        "समाप्त कर दिया जाएगा", "बंद केले जाईल", "काट दी जाएगी",
        "खाते बंद", "तात्काळ", "निलंबित केले जाईल", "विद्युत पुरवठा खंडित"
    };

    private static final String[] SENSITIVE_INFO_KEYWORDS = {
        "bank details", "bank account details", "passbook", "account number",
        "share your otp", "share the otp", "send your otp", "share your pin",
        "share your cvv", "aadhaar number", "pan number", "upi pin",
        "debit card number", "credit card number", "submit your bank",
        "submit your aadhaar", "send your bank", "share your bank",
        "share your details", "share your account", "provide your bank",
        "bank passbook", "enter your pin", "provide your otp",
        "बैंक विवरण", "पासबुक", "खाता संख्या", "आधार नंबर",
        "पैन नंबर", "ओटीपी साझा करें", "सीवीवी", "बँक तपशील", "खाते क्रमांक",
        "आधार क्रमांक", "ओटीपी शेअर करा", "पिन प्रविष्ट करा", "गुप्त क्रमांक"
    };

    private static final String[] REFUND_SCAM_KEYWORDS = {
        "accidentally sent", "sent by mistake", "by mistake", "wrong transfer",
        "wrongly transferred", "transferred to the wrong", "please refund",
        "refund it to this upi", "refund kar dijiye", "refund kijiye",
        "galti se bheja", "galti se transfer", "galti se", "galat transfer",
        "गलती से भेजा", "गलती से ट्रांसफर", "गलती से", "चुकून पाठवले", "चुकून ट्रान्सफर",
        "परत करा", "वापस भेजें", "वापस कर दीजिए"
    };

    private static final String[] CURRENCY_MARKERS = {"rs.", "rs ", "₹", "inr", "$", "eur", "usd", "£"};

    public static final String[] NUMERIC_FEATURES = {
        "char_len", "word_count", "digit_ratio", "exclaim_count", "special_ratio",
        "has_url", "has_phone", "currency_count", "urgency_count",
        "sensitive_info_count", "refund_scam_count"
    };

    public static final String[] CLASSES = {"PERSONAL", "TRANSACTIONAL", "PROMOTIONAL", "SCAM"};

    private final double isScamThreshold;
    private final Map<String, int[]> vocabIndexMap;
    private final Map<String, Double> vocabIdfMap;
    private final double[] numericMeans;
    private final double[] numericStds;
    private final double[][] weights;
    private final double[] bias;

    public AegisSmsClassifier(
        double isScamThreshold,
        Map<String, int[]> vocabIndexMap,
        Map<String, Double> vocabIdfMap,
        double[] numericMeans,
        double[] numericStds,
        double[][] weights,
        double[] bias
    ) {
        this.isScamThreshold = isScamThreshold;
        this.vocabIndexMap = vocabIndexMap;
        this.vocabIdfMap = vocabIdfMap;
        this.numericMeans = numericMeans;
        this.numericStds = numericStds;
        this.weights = weights;
        this.bias = bias;
    }

    public static class PredictionResult {
        public final String category;
        public final boolean isScam;
        public final double confidence;
        public final Map<String, Double> probabilities;

        public PredictionResult(String category, boolean isScam, double confidence, Map<String, Double> probabilities) {
            this.category = category;
            this.isScam = isScam;
            this.confidence = confidence;
            this.probabilities = probabilities;
        }
    }

    public static String urlwords(String url) {
        String u = url.toLowerCase();
        u = u.replaceFirst("^https?://", "");
        u = u.replaceFirst("^www\\.", "");
        String[] parts = u.split("[^a-z0-9]+");
        StringBuilder sb = new StringBuilder();
        for (String p : parts) {
            if (!p.isEmpty()) {
                if (sb.length() > 0) sb.append(" ");
                sb.append(p);
            }
        }
        return sb.toString();
    }

    public static Object[] cleanAndFeaturize(String text) {
        if (text == null) text = "";
        String t = Normalizer.normalize(text, Normalizer.Form.NFC);
        t = ZERO_WIDTH_RE.matcher(t).replaceAll("");
        int origLen = Math.max(1, t.length());

        // 1. Mask VPAs
        Matcher vpaMatcher = VPA_RE.matcher(t);
        List<String> vpas = new ArrayList<>();
        while (vpaMatcher.find()) {
            vpas.add(vpaMatcher.group());
        }
        String cleaned = t;
        for (String v : vpas) {
            cleaned = cleaned.replace(v, " upivpa ");
        }

        // 2. Extract and expand URLs
        Matcher urlMatcher = URL_RE.matcher(cleaned);
        List<String> urls = new ArrayList<>();
        while (urlMatcher.find()) {
            urls.add(urlMatcher.group());
        }
        double hasUrl = urls.isEmpty() ? 0.0 : 1.0;
        for (String u : urls) {
            cleaned = cleaned.replace(u, " " + urlwords(u) + " ");
        }

        // 3. Mask Phones
        Matcher phoneMatcher = PHONE_RE.matcher(cleaned);
        List<String> phones = new ArrayList<>();
        while (phoneMatcher.find()) {
            phones.add(phoneMatcher.group());
        }
        double hasPhone = phones.isEmpty() ? 0.0 : 1.0;
        for (String p : phones) {
            cleaned = cleaned.replace(p, " phonenumber ");
        }

        cleaned = cleaned.toLowerCase().replaceAll("\\s+", " ").trim();

        // 4. Extract word tokens
        List<String> words = new ArrayList<>();
        Matcher tokenMatcher = TOKEN_RE.matcher(cleaned);
        while (tokenMatcher.find()) {
            words.add(tokenMatcher.group());
        }
        double wordCount = words.size();

        int digitCount = 0;
        int exclaimCount = 0;
        int specialCount = 0;
        for (int i = 0; i < t.length(); i++) {
            char ch = t.charAt(i);
            if (Character.isDigit(ch)) digitCount++;
            if (ch == '!') exclaimCount++;
            if (!Character.isLetterOrDigit(ch) && !Character.isWhitespace(ch)) specialCount++;
        }
        double digitRatio = (double) digitCount / origLen;
        double specialRatio = (double) specialCount / origLen;

        String lowerFull = t.toLowerCase();
        double urgencyCount = 0;
        for (String k : URGENCY_KEYWORDS) {
            urgencyCount += countOccurrences(lowerFull, k.toLowerCase());
        }
        double currencyCount = 0;
        for (String k : CURRENCY_MARKERS) {
            currencyCount += countOccurrences(lowerFull, k.toLowerCase());
        }
        double sensitiveCount = 0;
        for (String k : SENSITIVE_INFO_KEYWORDS) {
            sensitiveCount += countOccurrences(lowerFull, k.toLowerCase());
        }
        double refundCount = 0;
        for (String k : REFUND_SCAM_KEYWORDS) {
            refundCount += countOccurrences(lowerFull, k.toLowerCase());
        }

        double[] numeric = new double[] {
            (double) origLen, wordCount, digitRatio, (double) exclaimCount, specialRatio,
            hasUrl, hasPhone, currencyCount, urgencyCount, sensitiveCount, refundCount
        };

        return new Object[] { cleaned, numeric, words };
    }

    private static int countOccurrences(String text, String sub) {
        int count = 0;
        int idx = 0;
        while ((idx = text.indexOf(sub, idx)) != -1) {
            count++;
            idx += sub.length();
        }
        return count;
    }

    public PredictionResult predict(String rawText) {
        Object[] pre = cleanAndFeaturize(rawText);
        String cleaned = (String) pre[0];
        double[] rawNumeric = (double[]) pre[1];
        @SuppressWarnings("unchecked")
        List<String> words = (List<String>) pre[2];

        Map<String, Integer> termCounts = new HashMap<>();
        int nWords = words.size();
        for (int n = 1; n <= 3; n++) {
            for (int i = 0; i <= nWords - n; i++) {
                StringBuilder sb = new StringBuilder();
                for (int j = i; j < i + n; j++) {
                    if (j > i) sb.append(" ");
                    sb.append(words.get(j));
                }
                String term = sb.toString();
                if (vocabIndexMap.containsKey(term)) {
                    termCounts.put(term, termCounts.getOrDefault(term, 0) + 1);
                }
            }
        }

        double l2Sum = 0.0;
        List<int[]> sparseIndices = new ArrayList<>();
        List<Double> sparseValues = new ArrayList<>();

        for (Map.Entry<String, Integer> e : termCounts.entrySet()) {
            String term = e.getKey();
            int cnt = e.getValue();
            int idx = vocabIndexMap.get(term)[0];
            double idf = vocabIdfMap.get(term);
            double tf = 1.0 + Math.log(cnt);
            double tfidf = tf * idf;
            sparseIndices.add(new int[] { idx });
            sparseValues.add(tfidf);
            l2Sum += tfidf * tfidf;
        }

        double l2Norm = l2Sum > 0.0 ? Math.sqrt(l2Sum) : 1.0;

        double[] logits = new double[4];
        for (int c = 0; c < 4; c++) {
            logits[c] = bias[c];
            for (int k = 0; k < sparseIndices.size(); k++) {
                int idx = sparseIndices.get(k)[0];
                double val = sparseValues.get(k) / l2Norm;
                logits[c] += weights[c][idx] * val;
            }
            for (int f = 0; f < rawNumeric.length; f++) {
                double scaled = (rawNumeric[f] - numericMeans[f]) / numericStds[f];
                int colIdx = vocabIndexMap.size() + f;
                logits[c] += weights[c][colIdx] * scaled;
            }
        }

        double maxLogit = Math.max(Math.max(logits[0], logits[1]), Math.max(logits[2], logits[3]));
        double sumExp = 0.0;
        double[] expLogits = new double[4];
        for (int c = 0; c < 4; c++) {
            expLogits[c] = Math.exp(logits[c] - maxLogit);
            sumExp += expLogits[c];
        }

        double[] probs = new double[4];
        int maxIdx = 0;
        for (int c = 0; c < 4; c++) {
            probs[c] = expLogits[c] / sumExp;
            if (probs[c] > probs[maxIdx]) maxIdx = c;
        }

        String predLabel = CLASSES[maxIdx];
        boolean isScam = (predLabel.equals("SCAM") || probs[3] >= isScamThreshold);

        Map<String, Double> probMap = new LinkedHashMap<>();
        for (int c = 0; c < 4; c++) {
            probMap.put(CLASSES[c], probs[c]);
        }

        return new PredictionResult(predLabel, isScam, probs[maxIdx], probMap);
    }
}
