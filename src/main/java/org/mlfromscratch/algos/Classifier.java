package org.mlfromscratch.algos;

public interface Classifier<T> {

	double[] train();

	T classify(double[] inputData);

}