package org.mlfromscratch.test;

import org.mlfromscratch.algos.Classifier;

public class BasicTester<O> {

	/**
	 * @return rate of error in classification
	 */
	public double test(Classifier<O> classifier, double[][] testData, O[] classes) {
		int correct = 0;
		
		for (int i = 0; i < testData.length; i++) {
			correct += classifier.classify(testData[i]) == classes[i] ? 1 : 0; 
		}
		
		return 1d - (double)correct / testData.length;
	}
	
}
