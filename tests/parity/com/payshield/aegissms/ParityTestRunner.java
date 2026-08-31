package com.payshield.aegissms;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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

        Matcher thMatcher = Pattern.compile("\"is_scam_operating_threshold\"\\s*:\\s*([0-9.eE+-]+)").matcher(contractJson);
        double threshold = thMatcher.find() ? Double.parseDouble(thMatcher.group(1)) : 0.50;

        Matcher meanMatcher = Pattern.compile("\"mean\"\\s*:\\s*\\[([^\\]]+)\\]").matcher(contractJson);
        Matcher stdMatcher = Pattern.compile("\"std\"\\s*:\\s*\\[([^\\]]+)\\]").matcher(contractJson);
        meanMatcher.find();
        stdMatcher.find();

        String[] meanStrs = meanMatcher.group(1).split(",");
        String[] stdStrs = stdMatcher.group(1).split(",");
        double[] numericMeans = new double[meanStrs.length];
        double[] numericStds = new double[stdStrs.length];
        for (int i = 0; i < meanStrs.length; i++) {
            numericMeans[i] = Double.parseDouble(meanStrs[i].trim());
            numericStds[i] = Double.parseDouble(stdStrs[i].trim());
        }

        Matcher biasMatcher = Pattern.compile("\"bias\"\\s*:\\s*\\[([^\\]]+)\\]").matcher(contractJson);
        biasMatcher.find();
        String[] biasStrs = biasMatcher.group(1).split(",");
        double[] bias = new double[4];
        for (int i = 0; i < 4; i++) bias[i] = Double.parseDouble(biasStrs[i].trim());

        Map<String, int[]> vocabIndexMap = new HashMap<>();
        Map<String, Double> vocabIdfMap = new HashMap<>();

        int vocabStart = contractJson.indexOf("\"vocabulary_idf\"");
        int weightsStart = contractJson.indexOf("\"weights\"");
        String vocabSection = contractJson.substring(vocabStart, weightsStart);

        Matcher itemMatcher = Pattern.compile("\"([^\"]+)\"\\s*:\\s*\\{\\s*\"i\"\\s*:\\s*(\\d+)\\s*,\\s*\"w\"\\s*:\\s*([0-9.eE+-]+)\\s*\\}").matcher(vocabSection);
        while (itemMatcher.find()) {
            String token = itemMatcher.group(1);
            int idx = Integer.parseInt(itemMatcher.group(2));
            double idf = Double.parseDouble(itemMatcher.group(3));
            vocabIndexMap.put(token, new int[]{idx});
            vocabIdfMap.put(token, idf);
        }

        double[][] weights = new double[4][];
        int pos = weightsStart;
        for (int r = 0; r < 4; r++) {
            int rowStart = contractJson.indexOf("[", pos);
            if (r == 0) {
                rowStart = contractJson.indexOf("[", rowStart + 1);
            }
            int rowEnd = contractJson.indexOf("]", rowStart);
            String rowStr = contractJson.substring(rowStart + 1, rowEnd);
            String[] wStrs = rowStr.split(",");
            double[] row = new double[wStrs.length];
            for (int j = 0; j < wStrs.length; j++) {
                row[j] = Double.parseDouble(wStrs[j].trim());
            }
            weights[r] = row;
            pos = rowEnd + 1;
        }

        System.out.println("Initialized AegisSmsClassifier with " + vocabIndexMap.size() + " tokens and " + weights[0].length + " weight dimensions.");
        AegisSmsClassifier classifier = new AegisSmsClassifier(threshold, vocabIndexMap, vocabIdfMap, numericMeans, numericStds, weights, bias);

        // Load 1000 golden vectors
        String goldenJson = new String(Files.readAllBytes(Paths.get(goldenPath)), "UTF-8");
        Pattern vecPattern = Pattern.compile(
            "\"vector_id\"\\s*:\\s*\"([^\"]+)\".*?\"raw_text\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\".*?\"python_probabilities\"\\s*:\\s*\\{([^}]+)\\}",
            Pattern.DOTALL
        );
        Matcher vecMatcher = vecPattern.matcher(goldenJson);

        int count = 0;
        double maxDelta = 0.0;
        int categoryMismatches = 0;
        int scamDecisionMismatches = 0;

        while (vecMatcher.find()) {
            count++;
            String vecId = vecMatcher.group(1);
            String rawText = vecMatcher.group(2).replace("\\\"", "\"").replace("\\n", "\n").replace("\\\\", "\\");
            String pyProbStr = vecMatcher.group(3);

            Map<String, Double> pyProbs = new HashMap<>();
            Matcher pMatcher = Pattern.compile("\"([A-Z]+)\"\\s*:\\s*([0-9.eE+-]+)").matcher(pyProbStr);
            while (pMatcher.find()) {
                pyProbs.put(pMatcher.group(1), Double.parseDouble(pMatcher.group(2)));
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

            double maxPyProb = -1.0;
            String maxPyClass = "UNKNOWN";
            for (Map.Entry<String, Double> e : pyProbs.entrySet()) {
                if (e.getValue() > maxPyProb) {
                    maxPyProb = e.getValue();
                    maxPyClass = e.getKey();
                }
            }

            if (!result.category.equals(maxPyClass)) {
                categoryMismatches++;
            }

            boolean pyIsScam = (maxPyClass.equals("SCAM") || pyProbs.getOrDefault("SCAM", 0.0) >= threshold);
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
