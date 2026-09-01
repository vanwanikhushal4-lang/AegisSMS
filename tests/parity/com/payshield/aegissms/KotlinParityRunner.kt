package com.payshield.aegissms

import java.io.File
import java.nio.file.Files
import java.nio.file.Paths
import org.json.JSONArray
import org.json.JSONObject

object KotlinParityRunner {

    @JvmStatic
    fun main(args: Array<String>) {
        val baseDir = if (args.isNotEmpty()) args[0] else "."
        val contractPath = Paths.get(baseDir, "artifacts", "aegis_model_contract.json").toString()
        val goldenPath = Paths.get(baseDir, "artifacts", "golden_parity_1000.json").toString()

        println("===============================================================")
        println("KOTLIN PARITY RUNNER (REAL JVM EXECUTION - ZERO SIMULATION)")
        println("===============================================================")
        println("Loading Contract: $contractPath")
        println("Loading Golden Vectors: $goldenPath")

        val contractJson = String(Files.readAllBytes(Paths.get(contractPath)), Charsets.UTF_8)
        val classifier = AegisSmsClassifier.fromContract(contractJson)

        println("Initialized AegisSmsClassifier with ${classifier.vocabIndexMap.size} tokens and ${classifier.weights[0].size} weight dimensions.")

        val goldenJson = String(Files.readAllBytes(Paths.get(goldenPath)), Charsets.UTF_8)
        val vecArray = JSONArray(goldenJson)

        var count = 0
        var maxDelta = 0.0
        var categoryMismatches = 0
        var scamDecisionMismatches = 0

        for (i in 0 until vecArray.length()) {
            count++
            val vecObj = vecArray.getJSONObject(i)
            val vecId = vecObj.getString("vector_id")
            val rawText = vecObj.getString("raw_text")
            val pyProbObj = vecObj.getJSONObject("python_probabilities")

            val pyProbs = mutableMapOf<String, Double>()
            for (c in AegisSmsClassifier.CLASSES) {
                pyProbs[c] = pyProbObj.getDouble(c)
            }

            val result = classifier.predict(rawText)

            var localMaxDelta = 0.0
            for (c in AegisSmsClassifier.CLASSES) {
                val pyVal = pyProbs[c] ?: 0.0
                val jvmVal = result.probabilities[c] ?: 0.0
                val delta = Math.abs(pyVal - jvmVal)
                if (delta > localMaxDelta) {
                    localMaxDelta = delta
                }
            }

            if (localMaxDelta > maxDelta) {
                maxDelta = localMaxDelta
            }

            val pyIsScam = (pyProbs["SCAM"] ?: 0.0) >= classifier.isScamThreshold
            val pyPredClass = if (pyIsScam) {
                "SCAM"
            } else {
                val nonScam = listOf("PERSONAL", "TRANSACTIONAL", "PROMOTIONAL")
                var maxC = nonScam[0]
                var maxV = pyProbs[maxC] ?: 0.0
                for (c in nonScam) {
                    val v = pyProbs[c] ?: 0.0
                    if (v > maxV) {
                        maxV = v
                        maxC = c
                    }
                }
                maxC
            }

            if (result.category != pyPredClass) {
                categoryMismatches++
            }

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
