package com.payshield.aegissms;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import org.json.JSONArray;
import org.json.JSONObject;

public class ParityTestRunner {

    public static void main(String[] args) throws Exception {
        String baseDir = args.length > 0 ? args[0] : ".";
        String contractPath = Paths.get(baseDir, "artifacts", "aegis_model_contract.json").toString();
        String goldenPath = Paths.get(baseDir, "artifacts", "golden_parity_1000.json").toString();

        System.out.println("===============================================================");
        System.out.println("JVM PARITY TEST RUNNER (REAL JAVA EXECUTION - ZERO SIMULATION)");
        System.out.println("===============================================================");
        System.out.println("Loading Contract: " + contractPath);
        System.out.println("Loading Golden Vectors: " + goldenPath);

        String contractJson = new String(Files.readAllBytes(Paths.get(contractPath)), "UTF-8");
        JSONObject root = new JSONObject(contractJson);

        double threshold = root.optDouble("is_scam_operating_threshold", 0.69);

        JSONArray meanArr = root.getJSONArray("numeric_means");
        JSONArray stdArr = root.getJSONArray("numeric_stds");
        double[] numericMeans = new double[meanArr.length()];
        double[] numericStds = new double[stdArr.length()];
        for (int i = 0; i < meanArr.length(); i++) {
            numericMeans[i] = meanArr.getDouble(i);
            numericStds[i] = stdArr.getDouble(i);
        }

        JSONArray biasArr = root.getJSONArray("bias");
        double[] bias = new double[biasArr.length()];
        for (int i = 0; i < biasArr.length(); i++) {
            bias[i] = biasArr.getDouble(i);
        }

        JSONArray weightsArr = root.getJSONArray("weights");
        double[][] weights = new double[weightsArr.length()][];
        for (int i = 0; i < weightsArr.length(); i++) {
            JSONArray rowArr = weightsArr.getJSONArray(i);
            double[] row = new double[rowArr.length()];
            for (int j = 0; j < rowArr.length(); j++) {
                row[j] = rowArr.getDouble(j);
            }
            weights[i] = row;
        }

        JSONObject vocabIndexObj = root.getJSONObject("vocab_index_map");
        JSONObject vocabIdfObj = root.getJSONObject("vocab_idf_map");

        Map<String, int[]> vocabIndexMap = new HashMap<>();
        Map<String, Double> vocabIdfMap = new HashMap<>();

        for (String term : vocabIndexObj.keySet()) {
            vocabIndexMap.put(term, new int[]{vocabIndexObj.getInt(term)});
            vocabIdfMap.put(term, vocabIdfObj.getDouble(term));
        }

        System.out.println("Initialized AegisSmsClassifier with " + vocabIndexMap.size() + " tokens and " + weights[0].length + " weight dimensions.");
        AegisSmsClassifier classifier = new AegisSmsClassifier(threshold, vocabIndexMap, vocabIdfMap, numericMeans, numericStds, weights, bias);

        // Load golden vectors
        String goldenJson = new String(Files.readAllBytes(Paths.get(goldenPath)), "UTF-8");
        JSONArray vecArray = new JSONArray(goldenJson);

        int count = 0;
        double maxDelta = 0.0;
        int categoryMismatches = 0;
        int scamDecisionMismatches = 0;

        for (int i = 0; i < vecArray.length(); i++) {
            count++;
            JSONObject vecObj = vecArray.getJSONObject(i);
            String vecId = vecObj.getString("vector_id");
            String rawText = vecObj.getString("raw_text");
            JSONObject pyProbObj = vecObj.getJSONObject("python_probabilities");

            Map<String, Double> pyProbs = new HashMap<>();
            for (String c : AegisSmsClassifier.CLASSES) {
                pyProbs.put(c, pyProbObj.getDouble(c));
            }

            AegisSmsClassifier.PredictionResult result = classifier.predict(rawText);

            double localMaxDelta = 0.0;
            for (String c : AegisSmsClassifier.CLASSES) {
                double pyVal = pyProbs.getOrDefault(c, 0.0);
                double jvmVal = result.probabilities.get(c);
                double delta = Math.abs(pyVal - jvmVal);
                if (delta > localMaxDelta) {
                    localMaxDelta = delta;
                }
            }

            if (localMaxDelta > maxDelta) {
                maxDelta = localMaxDelta;
            }

            boolean pyIsScam = pyProbs.getOrDefault("SCAM", 0.0) >= threshold;
            String pyPredClass;
            if (pyIsScam) {
                pyPredClass = "SCAM";
            } else {
                List<String> nonScam = Arrays.asList("PERSONAL", "TRANSACTIONAL", "PROMOTIONAL");
                String maxC = nonScam.get(0);
                double maxV = pyProbs.getOrDefault(maxC, 0.0);
                for (String c : nonScam) {
                    double v = pyProbs.getOrDefault(c, 0.0);
                    if (v > maxV) {
                        maxV = v;
                        maxC = c;
                    }
                }
                pyPredClass = maxC;
            }

            if (!result.category.equals(pyPredClass)) {
                categoryMismatches++;
            }

            if (result.isScam != pyIsScam) {
                scamDecisionMismatches++;
            }
        }

        System.out.println("\n---------------------------------------------------------------");
        System.out.println("PARITY EXECUTION RESULTS (1,000 VECTORS TESTED ON JVM):");
        System.out.println("  - Total Vectors Evaluated:      " + count);
        System.out.println("  - Max Floating Point Delta:     " + String.format("%.8e", maxDelta));
        System.out.println("  - Category Mismatches:          " + categoryMismatches);
        System.out.println("  - Scam Decision Mismatches:     " + scamDecisionMismatches);
        System.out.println("---------------------------------------------------------------");

        if (count >= 1000 && maxDelta < 1e-5 && categoryMismatches == 0 && scamDecisionMismatches == 0) {
            System.out.println("STATUS: [PASS] JVM PARITY CONTRACT FULLY VERIFIED (Delta < 1e-5, 0 Mismatches).");
            System.exit(0);
        } else {
            System.err.println("STATUS: [FAIL] Parity delta exceeded threshold!");
            System.exit(1);
        }
    }
}
