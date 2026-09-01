package com.payshield.aegissms

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.InputStream
import java.text.Normalizer
import java.util.regex.Matcher
import java.util.regex.Pattern
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Production AegisSMS 4-Way Intent & Threat Classifier for Android / Kotlin.
 * Zero external ML dependencies. Direct evaluation of TF-IDF n-grams + 11 numeric features.
 */
class AegisSmsClassifier(
    val isScamThreshold: Double,
    val vocabMap: Map<String, Pair<Int, Double>>, // token -> (index, idf)
    val numericMeans: DoubleArray,
    val numericStds: DoubleArray,
    val weights: Array<DoubleArray>, // [4][vocab_size + 11]
    val bias: DoubleArray // [4]
) {
    data class ParsedContract(
        val isScamThreshold: Double,
        val vocabMap: Map<String, Pair<Int, Double>>,
        val numericMeans: DoubleArray,
        val numericStds: DoubleArray,
        val weights: Array<DoubleArray>,
        val bias: DoubleArray
    )

    private constructor(p: ParsedContract) : this(p.isScamThreshold, p.vocabMap, p.numericMeans, p.numericStds, p.weights, p.bias)

    constructor(contractJson: String) : this(parseContract(JSONObject(contractJson)))

    constructor(inputStream: InputStream) : this(parseContract(JSONObject(inputStream.bufferedReader().use { it.readText() })))

    constructor(file: File) : this(parseContract(JSONObject(file.readText())))

    data class PredictionResult(
        val category: String,
        val isScam: Boolean,
        val confidence: Double,
        val probabilities: Map<String, Double>
    )

    fun predict(text: String): PredictionResult {
        val (cleanedText, rawNumeric, words) = cleanAndFeaturize(text)

        // 1. Extract 1-3 grams
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

        // 2. Compute sublinear TF-IDF and L2 norm
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

        // 3. Compute Logits
        val logits = bias.clone()
        val numFeaturesOffset = weights[0].size - 11

        for (c in 0 until 4) {
            for ((idx, v) in sparseEntries) {
                logits[c] += weights[c][idx] * (v / l2Norm)
            }
            for (f in rawNumeric.indices) {
                val scaled = (rawNumeric[f] - numericMeans[f]) / numericStds[f]
                val colIdx = numFeaturesOffset + f
                logits[c] += weights[c][colIdx] * scaled
            }
        }

        // 4. Softmax
        val maxLogit = logits.maxOrNull() ?: 0.0
        val expLogits = DoubleArray(4) { exp(logits[it] - maxLogit) }
        val sumExp = expLogits.sum()
        val probs = DoubleArray(4) { expLogits[it] / sumExp }

        // 5. Decision
        var maxIdx = 0
        for (i in 1 until 4) {
            if (probs[i] > probs[maxIdx]) maxIdx = i
        }

        val predLabel = CLASSES[maxIdx]
        val isScam = (predLabel == "SCAM" || probs[3] >= isScamThreshold)

        val probMap = LinkedHashMap<String, Double>()
        for (i in CLASSES.indices) {
            probMap[CLASSES[i]] = probs[i]
        }

        return PredictionResult(
            category = predLabel,
            isScam = isScam,
            confidence = probs[maxIdx],
            probabilities = probMap
        )
    }

    companion object {
        val CLASSES = arrayOf("PERSONAL", "TRANSACTIONAL", "PROMOTIONAL", "SCAM")

        private val ZERO_WIDTH_RE = Pattern.compile("[\u200B-\u200D\uFEFF\uFFFD\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]")
        private val VPA_RE = Pattern.compile("([a-zA-Z0-9.\\-_]{2,256}@[a-zA-Z]{2,64})", Pattern.CASE_INSENSITIVE)
        private val URL_RE = Pattern.compile("(https?://\\S+|www\\.\\S+|[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}(?:/\\S*)?)", Pattern.CASE_INSENSITIVE)
        private val PHONE_RE = Pattern.compile("(\\+?\\d[\\d\\- ]{7,}\\d)")
        private val CURRENCY_RE = Pattern.compile("(?:rs\\.?|inr|₹|\\$|£|eur)\\s*[\\d,]+(?:\\.\\d{1,2})?", Pattern.CASE_INSENSITIVE)
        private val URGENCY_RE = Pattern.compile(
            "\\b(urgent|immediately|action required|blocked|suspended|disconnect|cut off|expire|limited time|hours left|last chance|hurry|final notice|alert|warning|threat|coercive|विद्युत|तातडीने|लगेच|सावध|तुरंत|काट दिया)\\b",
            Pattern.CASE_INSENSITIVE
        )
        private val SENSITIVE_INFO_RE = Pattern.compile(
            "\\b(otp|pin|password|cvv|aadhaar|pan card|kyc|verify details|login to verify|account blocked|credit card|update kyc|केवायसी|पॅन कार्ड|आधार|ओटीपी|खाते ब्लॉक)\\b",
            Pattern.CASE_INSENSITIVE
        )
        private val REFUND_SCAM_RE = Pattern.compile(
            "\\b(wrong transfer|sent by mistake|galti se|refund|claim reward|lottery|kbc|lucky draw|won prize|won lakh|won crore|won cash|bheja hai|वापस भेजें|बक्षीस|लॉटरी)\\b",
            Pattern.CASE_INSENSITIVE
        )
        private val PUNCT_RE = Pattern.compile("[\\.,;:!\\?\\(\\)\\[\\]\\{\\}\"\'<>\\/\\-_+*~`@#\\$%\\^&\\\\|=]")

        fun countMatches(matcher: Matcher): Double {
            var c = 0
            while (matcher.find()) c++
            return c.toDouble()
        }

        fun cleanAndFeaturize(text: String): Triple<String, DoubleArray, List<String>> {
            var rawNorm = Normalizer.normalize(text ?: "", Normalizer.Form.NFC)
            rawNorm = ZERO_WIDTH_RE.matcher(rawNorm).replaceAll("")
            val charLen = rawNorm.length.toDouble()

            val hasUrl = if (URL_RE.matcher(rawNorm).find()) 1.0 else 0.0
            val hasPhone = if (PHONE_RE.matcher(rawNorm).find()) 1.0 else 0.0
            val currencyCount = countMatches(CURRENCY_RE.matcher(rawNorm))
            val urgencyCount = countMatches(URGENCY_RE.matcher(rawNorm))
            val sensitiveCount = countMatches(SENSITIVE_INFO_RE.matcher(rawNorm))
            val refundCount = countMatches(REFUND_SCAM_RE.matcher(rawNorm))

            var digitCount = 0
            var exclaimCount = 0.0
            var specialCount = 0
            for (ch in rawNorm) {
                if (ch.isDigit()) digitCount++
                if (ch == '!') exclaimCount++
                if (!ch.isLetterOrDigit() && !ch.isWhitespace()) specialCount++
            }

            val digitRatio = if (charLen > 0) digitCount / charLen else 0.0
            val specialRatio = if (charLen > 0) specialCount / charLen else 0.0

            var cleaned = rawNorm
            cleaned = VPA_RE.matcher(cleaned).replaceAll(" upivpa ")

            val urlMatcher = URL_RE.matcher(cleaned)
            val sb = StringBuffer()
            while (urlMatcher.find()) {
                val url = urlMatcher.group(0)
                val parts = url.split(Regex("[/._\\-:=?&%]"))
                val words = parts.filter { it.length > 1 && !it.matches(Regex("\\d+")) }.map { it.lowercase() }
                val repl = " httpurl " + words.joinToString(" ") + " "
                urlMatcher.appendReplacement(sb, Matcher.quoteReplacement(repl))
            }
            urlMatcher.appendTail(sb)
            cleaned = sb.toString()

            cleaned = PHONE_RE.matcher(cleaned).replaceAll(" phonenumber ")
            cleaned = PUNCT_RE.matcher(cleaned).replaceAll(" ")
            cleaned = cleaned.replace(Regex("\\s+"), " ").trim()

            val words = cleaned.lowercase().split(Regex("\\s+")).filter { it.isNotEmpty() }
            val wordCount = words.size.toDouble()

            val rawNumeric = doubleArrayOf(
                charLen,
                wordCount,
                digitRatio,
                exclaimCount,
                specialRatio,
                hasUrl,
                hasPhone,
                currencyCount,
                urgencyCount,
                sensitiveCount,
                refundCount
            )

            return Triple(cleaned, rawNumeric, words)
        }

        private fun parseContract(json: JSONObject): ParsedContract {
            val threshold = json.optDouble("is_scam_operating_threshold", 0.69)

            val normalizer = json.getJSONObject("feature_normalizer")
            val meanArr = normalizer.getJSONArray("mean")
            val stdArr = normalizer.getJSONArray("std")
            val means = DoubleArray(meanArr.length()) { meanArr.getDouble(it) }
            val stds = DoubleArray(stdArr.length()) { stdArr.getDouble(it) }

            val vocabObj = json.getJSONObject("vocabulary_idf")
            val vocab = HashMap<String, Pair<Int, Double>>()
            val keys = vocabObj.keys()
            while (keys.hasNext()) {
                val term = keys.next()
                val entry = vocabObj.getJSONObject(term)
                vocab[term] = Pair(entry.getInt("i"), entry.getDouble("w"))
            }

            val weightsArr = json.getJSONArray("weights")
            val nClasses = weightsArr.length()
            val weights = Array(nClasses) { c ->
                val row = weightsArr.getJSONArray(c)
                DoubleArray(row.length()) { row.getDouble(it) }
            }

            val biasArr = json.getJSONArray("bias")
            val bias = DoubleArray(biasArr.length()) { biasArr.getDouble(it) }

            return ParsedContract(
                isScamThreshold = threshold,
                vocabMap = vocab,
                numericMeans = means,
                numericStds = stds,
                weights = weights,
                bias = bias
            )
        }
    }
}
