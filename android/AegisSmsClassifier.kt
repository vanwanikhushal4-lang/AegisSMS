package com.payshield.aegissms

import java.io.File
import java.io.InputStream
import java.nio.charset.StandardCharsets
import java.text.Normalizer
import java.util.regex.Pattern
import org.json.JSONObject

/**
 * Production AegisSMS 4-Way Intent & Threat Classifier for Android / Kotlin.
 * Zero external ML dependencies. Direct evaluation of TF-IDF n-grams + 11 numeric features.
 */
class AegisSmsClassifier(
    val isScamThreshold: Double,
    val vocabIndexMap: Map<String, IntArray>,
    val vocabIdfMap: Map<String, Double>,
    val numericMeans: DoubleArray,
    val numericStds: DoubleArray,
    val weights: Array<DoubleArray>,
    val bias: DoubleArray
) {
    companion object {
        val CLASSES = arrayOf("PERSONAL", "TRANSACTIONAL", "PROMOTIONAL", "SCAM")

        private val ZERO_WIDTH_RE = Regex("[\u200B-\u200D\uFEFF\uFFFD\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f-\\x9f]")
        private val ACCOUNT_RE = Regex("\\b(?:a/c|ac|account\\s*ending\\s*in|account|card\\s*ending\\s*in|card\\s*ending|card|ending\\s*in|ending)\\s*(?:no\\.?|num|number)?\\s*[:#.]*\\s*([xX0-9]{3,18})\\b", RegexOption.IGNORE_CASE)
        private val VPA_RE = Regex("([a-zA-Z0-9.\\-_]{2,256}@[a-zA-Z]{2,64})", RegexOption.IGNORE_CASE)
        private val REF_RE = Regex("\\b(?:upi\\s*(?:reference|ref|txn)?|reference|ref|utr|awb|pnr|order|txn|rrn|crn|id)\\s*(?:no\\.?|num|number)?\\s*[:#.]*\\s*([a-zA-Z0-9]{6,20})\\b", RegexOption.IGNORE_CASE)
        private val URL_RE = Regex(
            "(https?://\\S+|www\\.\\S+|\\b[a-zA-Z0-9-]+\\.(?:com|org|net|in|co|co\\.in|gov|gov\\.in|edu|edu\\.in|io|ai|me|info|biz|link|site|top|xyz|club|live|shop|store|online|vip|app|apk|ly|gd|gl|cc|to|is|tv|uk|co\\.uk)(?:/[^\\s]*)?)",
            RegexOption.IGNORE_CASE
        )
        private val PHONE_RE = Regex("(\\+?\\d[\\d\\- ]{7,}\\d)")
        private val CURRENCY_RE = Regex("(?:rs\\.?|inr|₹|\\$|£|eur)\\s*[\\d,]+(?:\\.\\d{1,2})?", RegexOption.IGNORE_CASE)
        private val URGENCY_RE = Regex(
            "\\b(urgent|immediately|action required|avoid suspension|account.*locked|account.*blocked.*update|disconnect tonight|cut off tonight|expire.*hours|limited time|hours left|last chance|hurry|final notice|threat|coercive|विद्युत खंडित|तातडीने|लगेच कॉल|तुरंत कॉल|काट दिया)\\b",
            RegexOption.IGNORE_CASE
        )
        private val SENSITIVE_INFO_RE = Regex(
            "\\b(otp|pin|password|cvv|aadhaar|pan card|kyc|verify details|login to verify|credit card|update kyc|केवायसी|पॅन कार्ड|आधार|ओटीपी)\\b",
            RegexOption.IGNORE_CASE
        )
        private val REFUND_SCAM_RE = Regex(
            "\\b(wrong transfer|sent by mistake|galti se|refund|claim reward|lottery|kbc|lucky draw|won prize|won lakh|won crore|won cash|bheja hai|वापस भेजें|बक्षीस|लॉटरी)\\b",
            RegexOption.IGNORE_CASE
        )
        private val PUNCT_RE = Regex("[\\.,;:!\\?\\(\\)\\[\\]\\{\\}\"\'<>\\/\\-_+*~`@#\\$%\\^&\\\\|=]")

        fun fromContract(jsonString: String): AegisSmsClassifier {
            val root = JSONObject(jsonString)
            val threshold = root.optDouble("is_scam_operating_threshold", 0.69)
            val vIndexObj = root.getJSONObject("vocab_index_map")
            val vIdfObj = root.getJSONObject("vocab_idf_map")
            val nMeansArr = root.getJSONArray("numeric_means")
            val nStdsArr = root.getJSONArray("numeric_stds")
            val weightsArr = root.getJSONArray("weights")
            val biasArr = root.getJSONArray("bias")

            val vIndexMap = HashMap<String, IntArray>(vIndexObj.length())
            for (key in vIndexObj.keys()) {
                vIndexMap[key] = intArrayOf(vIndexObj.getInt(key))
            }

            val vIdfMap = HashMap<String, Double>(vIdfObj.length())
            for (key in vIdfObj.keys()) {
                vIdfMap[key] = vIdfObj.getDouble(key)
            }

            val nMeans = DoubleArray(nMeansArr.length()) { i -> nMeansArr.getDouble(i) }
            val nStds = DoubleArray(nStdsArr.length()) { i -> nStdsArr.getDouble(i) }
            val bias = DoubleArray(biasArr.length()) { i -> biasArr.getDouble(i) }

            val weights = Array(weightsArr.length()) { r ->
                val row = weightsArr.getJSONArray(r)
                DoubleArray(row.length()) { c -> row.getDouble(c) }
            }

            return AegisSmsClassifier(threshold, vIndexMap, vIdfMap, nMeans, nStds, weights, bias)
        }

        fun fromStream(stream: InputStream): AegisSmsClassifier {
            val jsonText = stream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
            return fromContract(jsonText)
        }

        fun fromFile(file: File): AegisSmsClassifier {
            return fromContract(file.readText(StandardCharsets.UTF_8))
        }

        fun deidentifyText(text: String): String {
            var t = Normalizer.normalize(text, Normalizer.Form.NFC)
            t = ZERO_WIDTH_RE.replace(t, "")
            t = Regex("\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b").replace(t, "<EMAIL>")
            t = VPA_RE.replace(t, "<VPA>")
            t = ACCOUNT_RE.replace(t, "A/c <ACCT>")
            t = REF_RE.replace(t, "Ref <REF>")
            t = PHONE_RE.replace(t, "<PHONE>")
            return t
        }

        fun cleanAndFeaturize(text: String): Triple<String, DoubleArray, List<String>> {
            val rawNorm = deidentifyText(text)
            val charLen = rawNorm.length.toDouble()

            val vpaMasked = VPA_RE.replace(rawNorm, " upivpa ")

            val hasUrl = if (URL_RE.containsMatchIn(vpaMasked)) 1.0 else 0.0
            val hasPhone = if (PHONE_RE.containsMatchIn(vpaMasked)) 1.0 else 0.0
            val currencyCount = CURRENCY_RE.findAll(rawNorm).count().toDouble()
            val urgencyCount = URGENCY_RE.findAll(rawNorm).count().toDouble()
            val sensitiveCount = SENSITIVE_INFO_RE.findAll(rawNorm).count().toDouble()
            val refundCount = REFUND_SCAM_RE.findAll(rawNorm).count().toDouble()

            var digitCount = 0
            var exclaimCount = 0.0
            var specialCount = 0
            for (ch in rawNorm) {
                if (ch.isDigit()) digitCount++
                if (ch == '!') exclaimCount += 1.0
                if (!ch.isLetterOrDigit() && !ch.isWhitespace()) specialCount++
            }

            val digitRatio = if (charLen > 0) digitCount / charLen else 0.0
            val specialRatio = if (charLen > 0) specialCount / charLen else 0.0

            var cleaned = vpaMasked

            cleaned = URL_RE.replace(cleaned) { match ->
                val url = match.value
                val parts = url.split(Regex("[/._\\-:=?&%]"))
                val words = parts.filter { it.length > 1 && !it.all { c -> c.isDigit() } }.map { it.lowercase() }
                " httpurl " + words.joinToString(" ") + " "
            }

            cleaned = PHONE_RE.replace(cleaned, " phonenumber ")
            cleaned = PUNCT_RE.replace(cleaned, " ")
            cleaned = cleaned.replace(Regex("\\s+"), " ").trim()

            val words = if (cleaned.isNotEmpty()) {
                cleaned.lowercase().split(Regex("\\s+")).filter { it.isNotEmpty() }
            } else {
                emptyList()
            }
            val wordCount = words.size.toDouble()

            val rawNumeric = doubleArrayOf(
                charLen, wordCount, digitRatio, exclaimCount, specialRatio,
                hasUrl, hasPhone, currencyCount, urgencyCount, sensitiveCount, refundCount
            )

            return Triple(cleaned, rawNumeric, words)
        }
    }

    data class PredictionResult(
        val category: String,
        val isScam: Boolean,
        val confidence: Double,
        val probabilities: Map<String, Double>
    )

    fun predict(text: String): PredictionResult {
        val (_, rawNumeric, words) = cleanAndFeaturize(text)

        // 1. Extract 1-3 grams
        val ngramCounts = HashMap<String, Int>()
        val nWords = words.size
        for (n in 1..3) {
            for (i in 0..(nWords - n)) {
                val sb = StringBuilder()
                for (j in 0 until n) {
                    if (j > 0) sb.append(" ")
                    sb.append(words[i + j])
                }
                val term = sb.toString()
                if (vocabIndexMap.containsKey(term)) {
                    ngramCounts[term] = (ngramCounts[term] ?: 0) + 1
                }
            }
        }

        // 2. Compute sublinear TF-IDF and L2 norm
        var l2Sum = 0.0
        val sparseEntries = ArrayList<Pair<Int, Double>>(ngramCounts.size)
        for ((term, count) in ngramCounts) {
            val idx = vocabIndexMap[term]!![0]
            val idf = vocabIdfMap[term]!!
            val tf = 1.0 + Math.log(count.toDouble())
            val tfidf = tf * idf
            sparseEntries.add(Pair(idx, tfidf))
            l2Sum += tfidf * tfidf
        }

        val l2Norm = if (l2Sum > 0.0) Math.sqrt(l2Sum) else 1.0

        // 3. Compute Logits
        val logits = bias.clone()
        val numFeaturesOffset = weights[0].size - 11

        for (c in 0 until 4) {
            for ((idx, tfidf) in sparseEntries) {
                val value = tfidf / l2Norm
                logits[c] += weights[c][idx] * value
            }
            for (f in rawNumeric.indices) {
                val scaled = (rawNumeric[f] - numericMeans[f]) / numericStds[f]
                val colIdx = numFeaturesOffset + f
                logits[c] += weights[c][colIdx] * scaled
            }
        }

        // 4. Softmax
        var maxLogit = logits[0]
        for (i in 1 until 4) {
            if (logits[i] > maxLogit) maxLogit = logits[i]
        }
        val expLogits = DoubleArray(4)
        var sumExp = 0.0
        for (i in 0 until 4) {
            expLogits[i] = Math.exp(logits[i] - maxLogit)
            sumExp += expLogits[i]
        }
        val probs = DoubleArray(4)
        for (i in 0 until 4) {
            probs[i] = expLogits[i] / sumExp
        }

        // 5. Calibrated Unified Decision
        val isScam = probs[3] >= isScamThreshold
        val predLabel = if (isScam) {
            "SCAM"
        } else {
            var nonScamMaxIdx = 0
            for (i in 1..2) {
                if (probs[i] > probs[nonScamMaxIdx]) nonScamMaxIdx = i
            }
            CLASSES[nonScamMaxIdx]
        }

        val probMap = LinkedHashMap<String, Double>()
        for (i in 0 until 4) {
            probMap[CLASSES[i]] = probs[i]
        }

        val confidence = if (isScam) probs[3] else probs[CLASSES.indexOf(predLabel)]
        return PredictionResult(predLabel, isScam, confidence, probMap)
    }
}
