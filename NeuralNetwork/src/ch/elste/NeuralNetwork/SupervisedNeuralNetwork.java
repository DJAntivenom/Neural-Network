package ch.elste.NeuralNetwork;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import ch.elste.NeuralNetwork.exceptions.IllegalWeightArgumentException;
import ch.elste.NeuralNetwork.exceptions.NoHiddenWeightException;

/**
 * The SurveilledNeuralNetwork represents an artificial neural network that is
 * trained by supervised learning and backpropagation.
 * 
 * @author Dillon
 * @version 1.0.4
 */
public class SupervisedNeuralNetwork {
	// Variables
	private int inputNodeSize, outputNodeSize;
	private int hiddenNodeSizes[];
	private int hiddenLayerCount;

	private static double tempDouble[];

	/**
	 * The learning rate is used in the feed-forward and backpropagation algorithms.
	 */
	private double learningRate;

	/**
	 * The error-rate is the average error of this network.
	 */
	private double errorRate;

	/**
	 * The current error of the network.
	 */
	private double currentCost;

	/**
	 * The total iterations the network went through.
	 */
	private int totalLearningIterations;

	private SimpleMatrix tempSimpleMatrix[];

	/**
	 * This array stores the derivatives of the calculations in a diagonal matrix
	 * for each layer.
	 */
	private SimpleMatrix derivatives[];

	/**
	 * Stores the deltas.
	 * <p>
	 * delta<sub>i</sub> =
	 * derivative<sub>i</sub>*weight<sub>i+1</sub>*delta<sub>i+1<sub>
	 */
	private SimpleMatrix deltas[];
	private SimpleMatrix weights[], biases[];

	// Constructor(s)

	/**
	 * Creates a new {@linkplain SupervisedNeuralNetwork} with given size.
	 * 
	 * @param inputNodeSize
	 *            the number of input nodes.
	 * @param outputNodeSize
	 *            the number of output nodes.
	 * @param hiddenNodeSizes
	 *            the number of hidden nodes.
	 * @since 0.0.1
	 */
	public SupervisedNeuralNetwork(double learningrate, int inputNodeSize, int outputNodeSize, int... hiddenNodeSizes) {
		if (hiddenNodeSizes.length <= 0) {
			throw new IllegalWeightArgumentException("There must be at least one hidden layer.");
		}

		this.inputNodeSize = inputNodeSize;
		this.outputNodeSize = outputNodeSize;
		this.hiddenNodeSizes = hiddenNodeSizes;

		hiddenLayerCount = hiddenNodeSizes.length;

		errorRate = 0d;
		currentCost = 0d;
		totalLearningIterations = 0;

		// init weights
		weights = new SimpleMatrix[hiddenLayerCount + 1];

		weights[0] = SimpleMatrix.random64(inputNodeSize, hiddenNodeSizes[0], -1, 1, new Random());

		for (int i = 0; i < hiddenLayerCount - 1; i++) {
			weights[i + 1] = SimpleMatrix.random64(hiddenNodeSizes[i], hiddenNodeSizes[i + 1], -1, 1, new Random());
		}

		weights[hiddenLayerCount] = SimpleMatrix.random64(hiddenNodeSizes[hiddenLayerCount - 1], outputNodeSize, -1, 1,
				new Random());

		// init biases
		biases = new SimpleMatrix[hiddenLayerCount + 1];
		biases[0] = SimpleMatrix.random64(1, hiddenNodeSizes[0], -1, 1, new Random());

		for (int i = 0; i < hiddenLayerCount - 1; i++) {
			biases[i + 1] = SimpleMatrix.random64(1, hiddenNodeSizes[i + 1], -1, 1, new Random());
		}

		biases[hiddenLayerCount] = SimpleMatrix.random64(1, outputNodeSize, -1, 1, new Random());

		this.learningRate = learningrate;
	}

	/**
	 * This Method calculates the outputs to given inputs using the current weights
	 * and biases.
	 * 
	 * @param inputs
	 *            the length of the array has to be the same as this network's
	 *            number of input nodes.
	 * @return a {@linkplain SimpleMatrix} with 1 row and
	 *         <code>outputNodeSize</code> columns.
	 * @see #feedForward(double[])
	 * @since 0.1.1
	 */
	public SimpleMatrix feedForward(SimpleMatrix inputs) {
		tempSimpleMatrix = new SimpleMatrix[weights.length];
		derivatives = new SimpleMatrix[weights.length];

		// Calculate values for first layer
		tempSimpleMatrix[0] = getSigmoid(getAppliedWeights(inputs, weights[0], biases[0]));
		derivatives[0] = getDiagSigmoidDerivative(tempSimpleMatrix[0]);

		// Calculate values for hidden layers
		for (int i = 0; i < hiddenLayerCount - 1; i++) {
			tempSimpleMatrix[i + 1] = getSigmoid(getAppliedWeights(tempSimpleMatrix[i], weights[i + 1], biases[i + 1]));
			derivatives[i + 1] = getDiagSigmoidDerivative(tempSimpleMatrix[i + 1]);
		}

		// Calculate Values for output layer
		tempSimpleMatrix[tempSimpleMatrix.length - 1] = getSigmoid(getAppliedWeights(
				tempSimpleMatrix[tempSimpleMatrix.length - 2], weights[weights.length - 1], biases[biases.length - 1]));
		derivatives[derivatives.length - 1] = getDiagSigmoidDerivative(tempSimpleMatrix[tempSimpleMatrix.length - 1]);

		// Return sigmoided values
		return tempSimpleMatrix[tempSimpleMatrix.length - 1];
	}

	/**
	 * This Method calculates the outputs to given inputs using the current weights
	 * and biases.
	 * 
	 * @param inputs
	 *            the length of the array has to be the same as this network's
	 *            number of input nodes.
	 * @return a double-array with the output values.
	 * @see #feedForward(SimpleMatrix)
	 * 
	 * @since 0.1.1
	 */
	public double[] feedForward(double inputs[]) {
		return feedForward(new SimpleMatrix(toHorizontalVectorArray(inputs))).getDDRM().data;
	}

	/**
	 * The backpropagation algorithm is one of the core mechanics of a neural
	 * network. Here all the weights and biases are changed according to the error
	 * of calculation using the current values.
	 * 
	 * @param inputs
	 *            the inputs for the neural network.
	 * @param expectedOutputs
	 *            the outputs the neural network should calculate.
	 * @return a double array containing the guess after updating weights
	 * 
	 * @since 0.2.1
	 */
	public double[] backpropagate(double inputs[], double expectedOutputs[]) {
		updateErrorRate(calcError(feedForward(inputs), expectedOutputs));
		updateCost(calcError(feedForward(inputs), expectedOutputs));

		// The changes to the weights are stored here
		SimpleMatrix[] deltaWeights = new SimpleMatrix[weights.length];

		// The input data stored as a SimpleMatrix
		SimpleMatrix inputMatrix = new SimpleMatrix(toHorizontalVectorArray(inputs));

		// The error of the output
		SimpleMatrix outputError = new SimpleMatrix(toVerticalVectorArray(calcError(feedForward(inputs), expectedOutputs)));

		deltas = new SimpleMatrix[weights.length];
		deltas[deltas.length - 1] = derivatives[derivatives.length - 1].mult(outputError);

		// deltaWeight = -learningrate*curr$\delta$*outputPrevLayer
		deltaWeights[weights.length - 1] = deltas[deltas.length - 1].scale(-learningRate).mult(tempSimpleMatrix[weights.length - 2]).transpose();

		for (int i = hiddenLayerCount - 1; i > 0; i--) {
			// $\delta$ = derivative*weights*prev$\delta$
			deltas[i] = derivatives[i].mult(weights[i + 1]).mult(deltas[i + 1]);

			// deltaWeight = -learningrate*curr$\delta$*outputPrevLayer
			deltaWeights[i] = deltas[i].scale(-learningRate).mult(tempSimpleMatrix[i - 1]).transpose();
		}
		
		deltas[0] = derivatives[0].mult(weights[1]).mult(deltas[1]);
		deltaWeights[0] = deltas[0].scale(-learningRate).mult(inputMatrix).transpose();

		// applying delta weights
		for (int i = 0; i < deltaWeights.length; i++) {
			weights[i] = weights[i].plus(deltaWeights[i]);
		}
		totalLearningIterations++;

		return feedForward(inputs);
	}

	/**
	 * Update the cost
	 * 
	 * @param error
	 *            current error
	 */
	private void updateCost(double[] error) {
		currentCost = calcCost(error);
	}

	/**
	 * Update the error rate
	 * 
	 * @param error
	 *            current error
	 */
	private void updateErrorRate(double[] error) {
		errorRate = (errorRate * totalLearningIterations + calcCost(error)) / (totalLearningIterations + 1);
	}

	/**
	 * This method applies the sigmoid-function to all values in m
	 * 
	 * @param m
	 *            the matrix containing the values.
	 * @return a new {@linkplain SimpleMatrix} with the sigmoided values.
	 * 
	 * @since 0.2.1
	 */
	public static SimpleMatrix getSigmoid(SimpleMatrix m) {
		return m.negative().elementExp().plus(1).elementPower(-1.0);
	}

	public static double[] getSigmoid(double d[]) {
		return getSigmoid(new SimpleMatrix(toHorizontalVectorArray(d))).getDDRM().data;
	}

	/**
	 * This method applies the derivative of the sigmoid-function to all values in m
	 * 
	 * @param m
	 *            the matrix containing the values.
	 * @return a new {@linkplain SimpleMatrix} with the values passed through the
	 *         derivative of sigmoid.
	 * 
	 * @since 0.2.1
	 */
	public static SimpleMatrix getSigmoidDerivative(SimpleMatrix m) {
		return m.elementMult(m.negative().plus(1d));
	}

	private static SimpleMatrix getDiagSigmoidDerivative(SimpleMatrix m) {
		return SimpleMatrix.diag(getSigmoidDerivative(m).getDDRM().data);
	}

	/**
	 * This method applies the derivative of the sigmoid-function to all values in m
	 * 
	 * @param d
	 *            the array containing the values.
	 * @return a double array with the values passed through the derivative of
	 *         sigmoid.
	 * 
	 * @since 0.2.1
	 */
	public static double[] getSigmoidDerivative(double d[]) {
		return getSigmoidDerivative(new SimpleMatrix(toHorizontalVectorArray(d))).getDDRM().data;
	}

	/**
	 * The method calculates the difference between the given inputs and the
	 * expected outputs.
	 * 
	 * @param inputs
	 *            the network's calculation.
	 * @param expectedOutputs
	 *            the true solutions.
	 * @return the error as a {@linkplain Double}-array.
	 * 
	 * @since 0.1.2
	 */
	public static double[] calcError(double inputs[], double expectedOutputs[]) {
		tempDouble = new double[inputs.length];
		for (int i = 0; i < tempDouble.length; i++) {
			tempDouble[i] = inputs[i] - expectedOutputs[i];
		}

		return tempDouble;
	}

	/**
	 * Return the current cost using the cost function:
	 * <p>
	 * {@code cost = 1/2*(error}<sub>{@code 0}</sub>{@code +...+error}<sub>{@code i}</sub>)
	 * </p>
	 * 
	 * @param inputs
	 * @return cost
	 */
	public static double calcCost(double inputs[]) {
		double temp = 0d;
		for (int i = 0; i < inputs.length; i++) {
			temp += 0.5 * Math.pow(inputs[i], 2);
		}

		return temp;
	}

	/**
	 * The method calculates the difference between the given inputs and the
	 * expected outputs.
	 * 
	 * @param inputs
	 *            the network's calculation.
	 * @param expectedOutputs
	 *            the true solutions.
	 * @return the error as a {@link SimpleMatrix}.
	 * 
	 * @since 0.1.2
	 */
	public static SimpleMatrix calcError(SimpleMatrix inputs, SimpleMatrix expectedOutputs) {
		return expectedOutputs.minus(inputs);
	}

	/**
	 * Creates a new 2-dimensional Array with {@code values.length} rows and 1
	 * column.
	 * 
	 * @param values
	 *            the data to transpose.
	 * @return the transposed array.
	 * 
	 * @since 0.1.1
	 */
	public static double[][] toVerticalVectorArray(double values[]) {
		double[][] data = new double[values.length][1];
		for (int i = 0; i < data.length; i++)
			data[i][0] = values[i];

		return data;
	}

	/**
	 * Creates a new 2-dimensional Array with {@code values.length} columns and 1
	 * row.
	 * 
	 * @param values
	 *            the data to transpose.
	 * @return the transposed array.
	 * 
	 * @since 1.0.4
	 */
	public static double[][] toHorizontalVectorArray(double values[]) {
		double[][] data = new double[1][values.length];
		data[0] = values;

		return data;
	}

	/**
	 * Calculate the sum of all the elements in a double-array.
	 * 
	 * @param ds
	 *            the array to be summed
	 * @return the sum as a double value
	 * 
	 * @since 1.0.1
	 */
	public static double getSum(double[] ds) {
		double sum = 0;
		for (double d : ds) {
			sum += d;
		}
		return sum;
	}

	/**
	 * The calculation of a single neuron.
	 * 
	 * @param inputs
	 *            the inputs or the calculations of the previous layer.
	 * @param weights
	 *            the weights of the current layer.
	 * @param biases
	 *            the biases of this layer.
	 * @return the calculations as a {@linkplain SimpleMatrix}.
	 * 
	 * @since 0.2.2
	 * @see SimpleMatrix
	 */
	private static SimpleMatrix getAppliedWeights(SimpleMatrix inputs, SimpleMatrix weights, SimpleMatrix biases) {
		return inputs.mult(weights).plus(biases);
	}

	/**
	 * A method to simplify the calculations for the backpropagation-algorithm.
	 * 
	 * @param error
	 *            the difference between calculation and expected solution.
	 * @param gradient
	 *            the derivative of the activation-function.
	 * @return a {@linkplain SimpleMatrix} holding the calculations.
	 * 
	 * @since 0.2.2
	 * @see SupervisedNeuralNetwork#backpropagate(double[], double[])
	 */
	private SimpleMatrix getDelta(SimpleMatrix error, SimpleMatrix gradient) {
		return gradient.elementMult(error).scale(learningRate);
	}

	/**
	 * A method to simplify the calculations for the backpropagation-algorithm.
	 * 
	 * @param error
	 *            the difference between calculation and expected solution.
	 * @param gradient
	 *            the derivative of the activation-function.
	 * @param values
	 *            the weights if needed.
	 * @return a {@linkplain SimpleMatrix} holding the calculations.
	 * 
	 * @since 0.2.2
	 * @see SupervisedNeuralNetwork#backpropagate(double[], double[])
	 */
	@SuppressWarnings("unused")
	@Deprecated
	private SimpleMatrix getDelta(SimpleMatrix error, SimpleMatrix gradient, SimpleMatrix values) {
		return getDelta(error, gradient).mult(values);
	}

	/**
	 * Prints information about the network to the console.
	 * 
	 * @deprecated
	 */
	@Deprecated
	public void print() {
		System.out.println("Weights input to hidden");
		weights[0].print();
		System.out.println("\n bias hidden");
		biases[0].print();
		System.out.println("\n weights hidden to output");
		weights[1].print();
		System.out.println("\n bias output");
		biases[1].print();
		System.out.println();
	}

	/**
	 * Returns the number of input-nodes.
	 * 
	 * @return amount of input-nodes
	 */
	public int getInputNodeSize() {
		return inputNodeSize;
	}

	/**
	 * Returns an array containing all sizes of the hidden layers.
	 * 
	 * @return the sizes of the hidden layers
	 */
	public int[] getHiddenNodeSizes() {
		return hiddenNodeSizes;
	}

	/**
	 * Returns the number of output-nodes
	 * 
	 * @return amount of output-nodes
	 */
	public int getOutputNodeSize() {
		return outputNodeSize;
	}

	/**
	 * Returns the learning-rate.
	 * 
	 * @return learning-rate
	 */
	public double getLearningrate() {
		return learningRate;
	}

	/**
	 * Set the network's {@link #learningRate learning-rate}.
	 * 
	 * @param learningRate
	 */
	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	/**
	 * Returns the {@link #errorRate}.
	 * 
	 * @return error-rate
	 */
	public double getErrorRate() {
		return errorRate;
	}

	/**
	 * Set the network's {@link #errorRate error-rate}.
	 * 
	 * @param errorRate
	 */
	public void setErrorRate(double errorRate) {
		this.errorRate = errorRate;
	}

	/**
	 * Returns the total amount of iterations the network went through.
	 * 
	 * @return amount of learning-iterations
	 */
	public int getTotalLearningIterations() {
		return totalLearningIterations;
	}

	/**
	 * Set {@link #totalLearningIterations}.
	 * 
	 * @param totalLearningIterations
	 */
	public void setTotalLearningIterations(int totalLearningIterations) {
		this.totalLearningIterations = totalLearningIterations;
	}

	/**
	 * Returns the {@link #currentCost}.
	 * 
	 * @return the cost
	 */
	public double getCost() {
		return currentCost;
	}

	/**
	 * Get the weights between the input and the hidden layer.
	 * 
	 * @return the weights as a double array
	 */
	public double[] getInputWeights() {
		return weights[0].getDDRM().data;
	}

	/**
	 * Set the weights between the input and the hidden layer.
	 * 
	 * @param weights
	 *            the weights as a double array
	 * @see #getInputWeights()
	 */
	public void setInputWeights(double[] weights) {
		this.weights[0].getDDRM().setData(weights);
	}

	/**
	 * Get the hidden weights, starting with the weights from hidden layer #1 to
	 * hidden layer #2
	 * 
	 * @return
	 *         <li>the hidden weights as an {@link java.util.ArrayList
	 *         <code>ArrayList&ltdouble[]&gt</code>}
	 *         <li><code>null</code> if there are no hidden weights
	 */
	public java.util.ArrayList<double[]> getHiddenWeights() {
		if (weights.length > 2) {
			java.util.ArrayList<double[]> temp = new java.util.ArrayList<>();
			for (int i = 1; i < weights.length - 1; i++)
				temp.add(weights[i].getDDRM().data);

			return temp;
		} else {
			return null;
		}
	}

	/**
	 * Set the hidden weights, starting with the weights from hidden layer #1 to
	 * hidden layer #2.
	 * 
	 * @param weights
	 *            the weights as an {@link java.util.ArrayList
	 *            <code>ArrayList&ltdouble[]&gt</code>}
	 * @throws NoHiddenWeightException
	 *             if the network was initialized with just one hidden layer
	 */
	public void setHiddenWeights(java.util.ArrayList<double[]> weights) throws NoHiddenWeightException {
		if (this.weights.length > 2) {
			for (int i = 0; i < weights.size() - 1; i++) {
				this.weights[i + 1].getDDRM().setData(weights.get(i));
			}
		} else {
			throw new NoHiddenWeightException();
		}
	}

	/**
	 * Get the weights between the last hidden and the output layer.
	 * 
	 * @return the weights as a double array
	 */
	public double[] getOutputWeights() {
		return weights[weights.length - 1].getDDRM().data;
	}

	/**
	 * Set the weights between the last hidden and the output layer.
	 * 
	 * @param weights
	 *            the weights as a double array
	 * 
	 * @see #getOutputWeights()
	 */
	public void setOutputWeights(double[] weights) {
		this.weights[weights.length - 1].getDDRM().data = weights;
	}
}