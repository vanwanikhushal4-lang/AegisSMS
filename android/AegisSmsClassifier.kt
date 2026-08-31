package com.payshield.aegissms

import java.io.InputStream
import java.text.Normalizer
import java.util.regex.Pattern
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt
import org.json.JSONObject

/**
 * Pure Kotlin on-device reference implementation of AegisSMS 4-Way Intent & Threat Engine.
 * Loads portable `aegis_model_contract.json` with zero third-party ML dependencies.
 */
class AegisSmsClassifier(contractJsonStream: InputStream) {

    private val classes: List<String>
    private val isScamThreshold: Double
    private val vocabMap: Map<String, Pair<Int, Double>> // token -> (index, idf)
    private val numericMeans: DoubleArray
    private val numericStds: DoubleArray
    private val weights: Array<DoubleArray> // [4][25011]
    private val bias: DoubleArray // [4]

    init {
        val jsonStr = contractJsonStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
        val root = JSONObject(jsonStr)

        val classesArr = root.getJSONArray("classes")
        classes = (0 until classesArr.length()).map { classesArr.getString(it) }
        isScamThreshold = root.getDouble("is_scam_operating_threshold")

        val vocabObj = root.getJSONObject("vocabulary_idf")
        val vMap = HashMap<String, Pair<Int, Double>>()
        val keys = vocabObj.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            val item = vocabObj.getJSONObject(key)
            vMap[key] = Pair(item.getInt("i"), item.getDouble("w"))
        }
        vocabMap = vMap

        val normObj = root.getJSONObject("feature_normalizer")
        val meanArr = normObj.getJSONArray("mean")
        numericMeans = DoubleArray(meanArr.length()) { meanArr.getDouble(it) }
        val stdArr = normObj.getJSONArray("std")
        numericStds = DoubleArray(stdArr.length()) { stdArr.getDouble(it) }

        val weightsArr = root.getJSONArray("weights")
        weights = Array(4) { c ->
            val row = weightsArr.getJSONArray(c)
            DoubleArray(row.length()) { row.getDouble(it) }
        }

        val biasArr = root.getJSONArray("bias")
        bias = DoubleArray(biasArr.length()) { biasArr.getDouble(it) }
    }

    data class PredictionResult(
        val category: String,
        val isScam: Boolean,
        val confidence: Double,
        val probabilities: Map<String, Double>
    )

    fun predict(rawText: String): PredictionResult {
        val (cleaned, rawNumeric, words) = cleanAndFeaturize(rawText)

        val ngramCounts = HashMap<String, Int>()
        val nWords = words.size
        for (n in 1..3) {
            for (i in 0..(nWords - n)) {
                val term = words.subList(i, i + n).joinToString(" ")
                if (vocabMap.containsKey(term)) {
                    ngramCounts[term] = (ngramCounts[term] ?: 0) + 1
                }
            }
        }

        var l2Sum = 0.0
        val sparseEntries = ArrayList<Pair<Int, Double>>()
        for ((term, count) in ngramCounts) {
            val meta = vocabMap[term]!!
            val tf = 1.0 + ln(count.toDouble())
            val tfidf = tf * meta.second
            sparseEntries.add(Pair(meta.first, tfidf))
            l2Sum += tfidf * tfidf
        }

        val l2Norm = if (l2Sum > 0.0) sqrt(l2Sum) else 1.0

        val logits = bias.clone()
        for (c in 0 until 4) {
            for ((idx, v) in sparseEntries) {
                logits[c] += weights[c][idx] * (v / l2Norm)
            }
            for (f in rawNumeric.indices) {
                val scaled = (rawNumeric[f] - numericMeans[f]) / numericStds[f]
                val colIdx = vocabMap.size + f
                logits[c] += weights[c][colIdx] * scaled
            }
        }

        val maxLogit = logits.maxOrNull() ?: 0.0
        val expLogits = DoubleArray(4) { exp(logits[it] - maxLogit) }
        val sumExp = expLogits.sum()
        val probs = DoubleArray(4) { expLogits[it] / sumExp }

        var maxIdx = 0
        for (i in 1 until 4) {
            if (probs[i] > probs[maxIdx]) maxIdx = i
        }

        val predLabel = classes[maxIdx]
        val isScam = (predLabel == "SCAM" || probs[3] >= isScamThreshold)

        val probMap = LinkedHashMap<String, Double>()
        for (i in classes.indices) {
            probMap[classes[i]] = probs[i]
        }

        return PredictionResult(
            category = predLabel,
            isScam = isScam,
            confidence = probs[maxIdx],
            probabilities = probMap
        )
    }

    companion object {
        private val VPA_RE = Pattern.compile("\\b[a-zA-Z0-9.\\-_]+@[a-zA-Z0-9.\\-_]+\\b")
        private val URL_RE = Pattern.compile(
            "(https?://\\S+|www\\.\\S+|(?:[a-zA-Z0-9-]+\\.)+(?:com|in|org|net|co|gov|edu|io|ai|xyz|top|site|online|apk|app|live|me|ly|link|info)(?:/\\S*)?)",
            Pattern.CASE_INSENSITIVE
        )
        private val PHONE_RE = Pattern.compile("(\\+?\\d[\\d\\- ]{7,}\\d)")
        private val ZERO_WIDTH_RE = Pattern.compile("[\\u200B-\\u200D\\uFEFF\\u200E\\u200F\\u00AD]")
        private val TOKEN_RE = Pattern.compile("[\\p{L}\\p{N}_]+")

        private val URGENCY_KEYWORDS = arrayOf(
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
        )

        private val SENSITIVE_INFO_KEYWORDS = arrayOf(
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
        )

        private val REFUND_SCAM_KEYWORDS = arrayOf(
            "accidentally sent", "sent by mistake", "by mistake", "wrong transfer",
            "wrongly transferred", "transferred to the wrong", "please refund",
            "refund it to this upi", "refund kar dijiye", "refund kijiye",
            "galti se bheja", "galti se transfer", "galti se", "galat transfer",
            "गलती से भेजा", "गलती से ट्रांसफर", "गलती से", "चुकून पाठवले", "चुकून ट्रान्सफर",
            "परत करा", "वापस भेजें", "वापस कर दीजिए"
        )

        private val CURRENCY_MARKERS = arrayOf("rs.", "rs ", "₹", "inr", "$", "eur", "usd", "£")

        fun urlwords(url: String): String {
            var u = url.lowercase()
            u = u.replaceFirst(Regex("^https?://"), "")
            u = u.replaceFirst(Regex("^www\\."), "")
            val parts = u.split(Regex("[^a-z0-9]+"))
            return parts.filter { it.isNotEmpty() }.joinToString(" ")
        }

        fun cleanAndFeaturize(text: String): Triple<String, DoubleArray, List<String>> {
            var t = Normalizer.normalize(text, Normalizer.Form.NFC)
            t = ZERO_WIDTH_RE.matcher(t).replaceAll("")
            val origLen = max(1, t.length)

            val vpaMatcher = VPA_RE.matcher(t)
            val vpas = ArrayList<String>()
            while (vpaMatcher.find()) vpas.add(vpaMatcher.group())
            var cleaned = t
            for (v in vpas) cleaned = cleaned.replace(v, " upivpa ")

            val urlMatcher = URL_RE.matcher(cleaned)
            val urls = ArrayList<String>()
            while (urlMatcher.find()) urls.add(urlMatcher.group())
            val hasUrl = if (urls.isEmpty()) 0.0 else 1.0
            for (u in urls) cleaned = cleaned.replace(u, " " + urlwords(u) + " ")

            val phoneMatcher = PHONE_RE.matcher(cleaned)
            val phones = ArrayList<String>()
            while (phoneMatcher.find()) phones.add(phoneMatcher.group())
            val hasPhone = if (phones.isEmpty()) 0.0 else 1.0
            for (p in phones) cleaned = cleaned.replace(p, " phonenumber ")

            cleaned = cleaned.lowercase().replace(Regex("\\s+"), " ").trim()

            val words = ArrayList<String>()
            val tokenMatcher = TOKEN_RE.matcher(cleaned)
            while (tokenMatcher.find()) words.add(tokenMatcher.group())
            val wordCount = words.size.toDouble()

            val digitCount = t.count { it.isDigit() }
            val exclaimCount = t.count { it == '!' }.toDouble()
            val specialCount = t.count { !it.isLetterOrDigit() && !it.isWhitespace() }
            val digitRatio = digitCount.toDouble() / origLen
            val specialRatio = specialCount.toDouble() / origLen

            val lowerFull = t.lowercase()
            val urgencyCount = URGENCY_KEYWORDS.sumOf { countOccurrences(lowerFull, it.lowercase()) }.toDouble()
            val currencyCount = CURRENCY_MARKERS.sumOf { countOccurrences(lowerFull, it.lowercase()) }.toDouble()
            val sensitiveCount = SENSITIVE_INFO_KEYWORDS.sumOf { countOccurrences(lowerFull, it.lowercase()) }.toDouble()
            val refundCount = REFUND_SCAM_KEYWORDS.sumOf { countOccurrences(lowerFull, it.lowercase()) }.toDouble()

            val numeric = doubleArrayOf(
                origLen.toDouble(), wordCount, digitRatio, exclaimCount, specialRatio,
                hasUrl, hasPhone, currencyCount, urgencyCount, sensitiveCount, refundCount
            )

            return Triple(cleaned, numeric, words)
        }

        private fun countOccurrences(text: String, sub: String): Int {
            var count = 0
            var idx = 0
            while (text.indexOf(sub, idx).also { idx = it } != -1) {
                count++
                idx += sub.length
            }
            return count
        }
    }
}
