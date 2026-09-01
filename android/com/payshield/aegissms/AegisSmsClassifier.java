package com.payshield.aegissms;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.Normalizer;
import java.util.*;
import java.util.regex.*;

/**
 * Production AegisSMS 4-Way Intent & Threat Classifier for Java / Android.
 * Zero external ML dependencies. Direct evaluation of TF-IDF n-grams + 11 numeric features.
 */
public class AegisSmsClassifier {

    public static final String[] CLASSES = new String[]{"PERSONAL", "TRANSACTIONAL", "PROMOTIONAL", "SCAM"};

    private static final Pattern ZERO_WIDTH_RE = Pattern.compile("[\u200B-\u200D\uFEFF\uFFFD\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]");
    private static final Pattern VPA_RE = Pattern.compile("([a-zA-Z0-9.\\-_]{2,256}@[a-zA-Z]{2,64})", Pattern.CASE_INSENSITIVE);
    private static final Pattern URL_RE = Pattern.compile("(https?://\\S+|www\\.\\S+|[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}(?:/\\S*)?)", Pattern.CASE_INSENSITIVE);
    private static final Pattern PHONE_RE = Pattern.compile("(\\+?\\d[\\d\\- ]{7,}\\d)");
    private static final Pattern CURRENCY_RE = Pattern.compile("(?:rs\\.?|inr|₹|\\$|£|eur)\\s*[\\d,]+(?:\\.\\d{1,2})?", Pattern.CASE_INSENSITIVE);
    private static final Pattern URGENCY_RE = Pattern.compile(
        "\\b(urgent|immediately|action required|blocked|suspended|disconnect|cut off|expire|limited time|hours left|last chance|hurry|final notice|alert|warning|threat|coercive|विद्युत|तातडीने|लगेच|सावध|तुरंत|काट दिया)\\b",
        Pattern.CASE_INSENSITIVE
    );
    private static final Pattern SENSITIVE_INFO_RE = Pattern.compile(
        "\\b(otp|pin|password|cvv|aadhaar|pan card|kyc|verify details|login to verify|account blocked|credit card|update kyc|केवायसी|पॅन कार्ड|आधार|ओटीपी|खाते ब्लॉक)\\b",
        Pattern.CASE_INSENSITIVE
    );
    private static final Pattern REFUND_SCAM_RE = Pattern.compile(
        "\\b(wrong transfer|sent by mistake|galti se|refund|claim reward|lottery|kbc|lucky draw|won prize|won lakh|won crore|won cash|bheja hai|वापस भेजें|बक्षीस|लॉटरी)\\b",
        Pattern.CASE_INSENSITIVE
    );
    private static final Pattern PUNCT_RE = Pattern.compile("[\\.,;:!\\?\\(\\)\\[\\]\\{\\}\"\'<>\\/\\-_+*~`@#\\$%\\^&\\\\|=]");

    public final double isScamThreshold;
    public final Map<String, int[]> vocabIndexMap;
    public final Map<String, Double> vocabIdfMap;
    public final double[] numericMeans;
    public final double[] numericStds;
    public final double[][] weights;
    public final double[] bias;

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

    private static double countMatches(Matcher matcher) {
        int count = 0;
        while (matcher.find()) count++;
        return (double) count;
    }

    public static Object[] cleanAndFeaturize(String text) {
        String rawNorm = Normalizer.normalize(text != null ? text : "", Normalizer.Form.NFC);
        rawNorm = ZERO_WIDTH_RE.matcher(rawNorm).replaceAll("");
        double charLen = (double) rawNorm.length();

        double hasUrl = URL_RE.matcher(rawNorm).find() ? 1.0 : 0.0;
        double hasPhone = PHONE_RE.matcher(rawNorm).find() ? 1.0 : 0.0;
        double currencyCount = countMatches(CURRENCY_RE.matcher(rawNorm));
        double urgencyCount = countMatches(URGENCY_RE.matcher(rawNorm));
        double sensitiveCount = countMatches(SENSITIVE_INFO_RE.matcher(rawNorm));
        double refundCount = countMatches(REFUND_SCAM_RE.matcher(rawNorm));

        int digitCount = 0;
        double exclaimCount = 0.0;
        int specialCount = 0;
        for (int i = 0; i < rawNorm.length(); i++) {
            char ch = rawNorm.charAt(i);
            if (Character.isDigit(ch)) digitCount++;
            if (ch == '!') exclaimCount++;
            if (!Character.isLetterOrDigit(ch) && !Character.isWhitespace(ch)) specialCount++;
        }

        double digitRatio = charLen > 0 ? (double) digitCount / charLen : 0.0;
        double specialRatio = charLen > 0 ? (double) specialCount / charLen : 0.0;

        String cleaned = rawNorm;
        cleaned = VPA_RE.matcher(cleaned).replaceAll(" upivpa ");

        Matcher urlMatcher = URL_RE.matcher(cleaned);
        StringBuffer sb = new StringBuffer();
        while (urlMatcher.find()) {
            String url = urlMatcher.group(0);
            String[] parts = url.split("[/._\\-:=?&%]");
            StringBuilder wordsBuilder = new StringBuilder();
            for (String p : parts) {
                if (p.length() > 1 && !p.matches("\\d+")) {
                    if (wordsBuilder.length() > 0) wordsBuilder.append(" ");
                    wordsBuilder.append(p.toLowerCase());
                }
            }
            String repl = " httpurl " + wordsBuilder.toString() + " ";
            urlMatcher.appendReplacement(sb, Matcher.quoteReplacement(repl));
        }
        urlMatcher.appendTail(sb);
        cleaned = sb.toString();

        cleaned = PHONE_RE.matcher(cleaned).replaceAll(" phonenumber ");
        cleaned = PUNCT_RE.matcher(cleaned).replaceAll(" ");
        cleaned = cleaned.replaceAll("\\s+", " ").trim();

        List<String> words = new ArrayList<>();
        if (!cleaned.isEmpty()) {
            for (String w : cleaned.toLowerCase().split("\\s+")) {
                if (!w.isEmpty()) words.add(w);
            }
        }
        double wordCount = (double) words.size();

        double[] rawNumeric = new double[]{
            charLen, wordCount, digitRatio, exclaimCount, specialRatio,
            hasUrl, hasPhone, currencyCount, urgencyCount, sensitiveCount, refundCount
        };

        return new Object[]{cleaned, rawNumeric, words};
    }

    public PredictionResult predict(String text) {
        Object[] cf = cleanAndFeaturize(text);
        String cleanedText = (String) cf[0];
        double[] rawNumeric = (double[]) cf[1];
        @SuppressWarnings("unchecked")
        List<String> words = (List<String>) cf[2];

        // 1. Extract 1-3 grams
        Map<String, Integer> ngramCounts = new HashMap<>();
        int nWords = words.size();
        for (int n = 1; n <= 3; n++) {
            for (int i = 0; i <= nWords - n; i++) {
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j < n; j++) {
                    if (j > 0) sb.append(" ");
                    sb.append(words.get(i + j));
                }
                String term = sb.toString();
                if (vocabIndexMap.containsKey(term)) {
                    ngramCounts.put(term, ngramCounts.getOrDefault(term, 0) + 1);
                }
            }
        }

        // 2. Compute sublinear TF-IDF and L2 norm
        double l2Sum = 0.0;
        List<Map.Entry<Integer, Double>> sparseEntries = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : ngramCounts.entrySet()) {
            String term = entry.getKey();
            int count = entry.getValue();
            int idx = vocabIndexMap.get(term)[0];
            double idf = vocabIdfMap.get(term);
            double tf = 1.0 + Math.log((double) count);
            double tfidf = tf * idf;
            sparseEntries.add(new AbstractMap.SimpleEntry<>(idx, tfidf));
            l2Sum += tfidf * tfidf;
        }

        double l2Norm = l2Sum > 0.0 ? Math.sqrt(l2Sum) : 1.0;

        // 3. Compute Logits
        double[] logits = bias.clone();
        int numFeaturesOffset = weights[0].length - 11;

        for (int c = 0; c < 4; c++) {
            for (Map.Entry<Integer, Double> entry : sparseEntries) {
                int idx = entry.getKey();
                double val = entry.getValue() / l2Norm;
                logits[c] += weights[c][idx] * val;
            }
            for (int f = 0; f < rawNumeric.length; f++) {
                double scaled = (rawNumeric[f] - numericMeans[f]) / numericStds[f];
                int colIdx = numFeaturesOffset + f;
                logits[c] += weights[c][colIdx] * scaled;
            }
        }

        // 4. Softmax
        double maxLogit = logits[0];
        for (int i = 1; i < 4; i++) {
            if (logits[i] > maxLogit) maxLogit = logits[i];
        }
        double[] expLogits = new double[4];
        double sumExp = 0.0;
        for (int i = 0; i < 4; i++) {
            expLogits[i] = Math.exp(logits[i] - maxLogit);
            sumExp += expLogits[i];
        }
        double[] probs = new double[4];
        for (int i = 0; i < 4; i++) {
            probs[i] = expLogits[i] / sumExp;
        }

        // 5. Decision
        int maxIdx = 0;
        for (int i = 1; i < 4; i++) {
            if (probs[i] > probs[maxIdx]) maxIdx = i;
        }

        String predLabel = CLASSES[maxIdx];
        boolean isScam = predLabel.equals("SCAM") || probs[3] >= isScamThreshold;

        Map<String, Double> probMap = new LinkedHashMap<>();
        for (int i = 0; i < 4; i++) {
            probMap.put(CLASSES[i], probs[i]);
        }

        return new PredictionResult(predLabel, isScam, probs[maxIdx], probMap);
    }
}
