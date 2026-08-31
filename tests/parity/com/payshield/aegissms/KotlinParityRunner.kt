package com.payshield.aegissms

import java.io.File
import java.io.FileInputStream
import java.nio.file.Files
import java.nio.file.Paths
import kotlin.math.abs
import org.json.JSONArray
import org.json.JSONObject

object KotlinParityRunner {

    @JvmStatic
    fun main(args: Array<String>) {
        val baseDir = if (args.isNotEmpty()) args[0] else "."
        val contractPath = Paths.get(baseDir, "artifacts", "aegis_model_contract.json").toString()
        val goldenPath = Paths.get(baseDir, "artifacts", "golden_parity_1000.json").toString()

        println("===============================================================")
        println("KOTLIN ON-DEVICE CLASSIFIER PARITY TEST RUNNER (REAL KOTLIN)")
        println("===============================================================")
        println("Loading Android Contract via Kotlin: $contractPath")
        println("Loading Golden Vectors: $goldenPath")

        val contractFile = File(contractPath)
        val classifier = AegisSmsClassifier(FileInputStream(contractFile))

        val goldenJsonStr = String(Files.readAllBytes(Paths.get(goldenPath)), Charsets.UTF_8)
        val goldenArray = JSONArray(goldenJsonStr)

        var count = 0
        var maxDelta = 0.0
        var categoryMismatches = 0
        var scamDecisionMismatches = 0

        for (i in 0 until goldenArray.length()) {
            count++
            val item = goldenArray.getJSONObject(i)
            val vecId = item.getString("vector_id")
            val rawText = item.getString("raw_text")
            val pyProbsObj = item.getJSONObject("python_probabilities")

            val pyProbs = HashMap<String, Double>()
            val keys = pyProbsObj.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                pyProbs[k] = pyProbsObj.getDouble(k)
            }

            val result = classifier.predict(rawText)

            var localMaxDelta = 0.0
            for (c in listOf("PERSONAL", "TRANSACTIONAL", "PROMOTIONAL", "SCAM")) {
                val pyVal = pyProbs[c] ?: 0.0
                val ktVal = result.probabilities[c] ?: 0.0
                val delta = abs(pyVal - ktVal)
                if (delta > localMaxDelta) {
                    localMaxDelta = delta
                }
            }

            if (localMaxDelta > maxDelta) {
                maxDelta = localMaxDelta
            }

            var maxPyProb = -1.0
            var maxPyClass = "UNKNOWN"
            for ((k, v) in pyProbs) {
                if (v > maxPyProb) {
                    maxPyProb = v
                    maxPyClass = k
                }
            }

            if (result.category != maxPyClass) {
                categoryMismatches++
            }

            // Calibrated production is_scam rule
            val pyIsScam = (maxPyClass == "SCAM" || (pyProbs["SCAM"] ?: 0.0) >= 0.69)
            if (result.isScam != pyIsScam) {
                scamDecisionMismatches++
            }
        }

        println("\n---------------------------------------------------------------")
        println("KOTLIN PARITY EXECUTION RESULTS (1,000 VECTORS TESTED ON JVM):")
        println("  - Total Vectors Evaluated:      $count")
        println("  - Max Floating Point Delta:     " + String.format("%.8e", maxDelta))
        println("  - Category Mismatches:          $categoryMismatches")
        println("  - Scam Decision Mismatches:     $scamDecisionMismatches")
        println("---------------------------------------------------------------")

        if (count >= 1000 && maxDelta < 1e-5 && categoryMismatches == 0 && scamDecisionMismatches == 0) {
            println("STATUS: [PASS] KOTLIN PARITY CONTRACT FULLY VERIFIED (Delta < 1e-5, 0 Mismatches).")
            System.exit(0)
        } else {
            System.err.println("STATUS: [FAIL] Kotlin parity delta exceeded threshold or mismatches found!")
            System.exit(1)
        }
    }
}
