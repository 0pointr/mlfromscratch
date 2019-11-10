package org.mlfromscratch.demo;

import java.util.Arrays;
import java.util.stream.IntStream;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.mlfromscratch.algos.Classifier;
import org.mlfromscratch.algos.SingleLayerPerceptron;
import org.mlfromscratch.test.BasicTester;

public class SLPDemo {
	
	private void test1(String testName) {
		
		System.out.println("Running " + testName);
		
		/**
		 * Fit to the binary OR function
		 */
		double[][] inputData = {
				{0,0}, {0,1}, {1,0}, {1,1}	
		};
		int outputClass[] = {-1, 1, 1, 1};
		double maxError = 0d;
		int epochs = 100;
		
		Classifier<Integer> slp = new SingleLayerPerceptron(2, 4, inputData, outputClass, maxError, epochs);
		
		double[] w = slp.train();
		
		System.out.println(Arrays.toString(w));
		
		double error = new BasicTester<Integer>()
							.test(slp, inputData, IntStream.of(outputClass).boxed().toArray(Integer[]::new));
		
		System.out.println("Test error: " + error);
	}
	
	private void test2(String testName) {
		
		System.out.println("Running " + testName);
		
		int inputSize = 500;
		NormalDistribution nd1 = new NormalDistribution(-2d, 1d);
		NormalDistribution nd2 = new NormalDistribution(2d, 1d);
		int[] outputClasses = new int[inputSize];
		
		double inputData[][] = new double[inputSize][2];
		for (int i = 0; i < inputSize / 2; i++) {
			inputData[i][0] = nd1.sample();
			inputData[i][1] = nd2.sample();
			outputClasses[i] = 1;
		}
		int offset = inputSize / 2;
		for (int i = offset; i < inputSize; i++) {
			inputData[i][0] = nd2.sample();
			inputData[i][1] = nd1.sample();
			outputClasses[i] = -1;
		}
		
		Classifier<Integer> slp = new SingleLayerPerceptron(2, inputSize, inputData, outputClasses, 0.1, 500);
		
		double[] w = slp.train();
		
		System.out.println(Arrays.toString(w));
		
		double error = new BasicTester<Integer>()
							.test(slp, inputData, IntStream.of(outputClasses).boxed().toArray(Integer[]::new));
		
		System.out.println("Test error: " + error);
	}
	
	public static void main(String[] args) {
		new SLPDemo().test1("Fit to OR function");
		new SLPDemo().test2("Fit to random data from 2 normal distributions: (m:-2, sd:1):1, (m:2, sd:1):-1");
	}

}
