package org.mlfromscratch.algos;

import java.util.Arrays;

public class SingleLayerPerceptron implements Classifier<Integer> {

	int dimensions;
	int inputSize;
	
	double[][] inputData;
	int[] outputClass;
	
	double[] weights;
	
	double maxError;
	int epochs;
	
	public SingleLayerPerceptron(int dimension, int inputSize, double[][] inputData, int[] outputClass, double maxError, int epochs) {
		super();
		this.dimensions = dimension;
		this.inputSize = inputSize;
		this.inputData = inputData;
		this.outputClass = outputClass;
		this.maxError = maxError;
		this.epochs = epochs;
		this.weights = new double[dimension + 1];
		
		Arrays.fill(this.weights, 0d);
	}
	
	@Override
	public double[] train() {
		for (int i = 0; i < epochs; i++) {
			
			int correctClassifications = 0;
			
			for (int j = 0; j < inputSize; j++) {
				
				int classification = classify(inputData[j]);
				
				if (outputClass[j] >= 0 && classification < 0) {
					weights[0] += 1;						// x0 = 1
					for (int d = 1; d < dimensions + 1; d++) {
						weights[d] += inputData[j][d-1];
					}
				} else if (outputClass[j] < 0 && classification >= 0) {
					weights[0] -= 1;						// x0 = 1
					for (int d = 1; d < dimensions + 1; d++) {
						weights[d] -= inputData[j][d-1];
					}
				}
				else
					correctClassifications += 1;
				
			}
			
			double error = 1d - (double)correctClassifications / inputSize;
			
			if (error <= maxError)
				break;
		}
		
		return this.weights;
	}
	
	@Override
	public Integer classify(double[] inputData) {
		double dotProduct = this.weights[0];		// w0 * 1 = bias or threshold
		
		for (int d = 1; d < dimensions + 1; d++) {
			dotProduct += inputData[d - 1] * weights[d];
		}
		
		return dotProduct > 0 ? 1 : -1;
	}
	
}
