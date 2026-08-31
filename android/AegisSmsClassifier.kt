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
        val jsonStr = contractJsonStream.bufferedReader().use { it.readText() }
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
        val (cleaned, rawNumeric) = preprocess(rawText)
        val words = extractTokens(cleaned)

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

        val probMap = HashMap<String, Double>()
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

    private fun preprocess(text: String): Pair<String, DoubleArray> {
        var t = Normalizer.normalize(text, Normalizer.Form.NFC)
        t = t.replace(Regex("[\u200B-\u200D\uFEFF\u200E\u200F\u00AD]"), "")
        val origLen = max(1, t.length)

        val urlMatches = Pattern.compile("(https?://\\S+|www\\.\\S+|[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}(?:/\\S*)?)", Pattern.CASE_INSENSITIVE).matcher(t)
        val hasUrl = if (urlMatches.find()) 1.0 else 0.0

        val phoneMatches = Pattern.compile("(\\+?\\d[\\d\\-\\s]{8,}\\d)").matcher(t)
        val hasPhone = if (phoneMatches.find()) 1.0 else 0.0

        val digits = t.count { it.isDigit() }
        val digitRatio = digits.toDouble() / origLen
        val exclaims = t.count { it == '!' }.toDouble()
        val specials = t.count { !it.isLetterOrDigit() && !it.isWhitespace() }
        val specialRatio = specials.toDouble() / origLen

        val cleaned = t.lowercase().replace(Regex("\\s+"), " ").trim()
        val words = extractTokens(cleaned)

        val numericFeats = doubleArrayOf(
            origLen.toDouble(), words.size.toDouble(), digitRatio, exclaims, specialRatio,
            hasUrl, hasPhone, 0.0, 0.0, 0.0, 0.0
        )
        return Pair(cleaned, numericFeats)
    }

    private fun extractTokens(text: String): List<String> {
        val matcher = Pattern.compile("(?u)\\b\\w+\\b").matcher(text)
        val tokens = ArrayList<String>()
        while (matcher.find()) {
            tokens.add(matcher.group())
        }
        return tokens
    }
}
