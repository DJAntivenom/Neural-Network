package ch.elste.NeuralNetwork.exceptions;

/**
 * Thrown if hidden weights are trying to be accessed but there are none.
 * @author Dillon Elste
 *
 */
public class NoHiddenWeightException extends Exception {
	private static final long serialVersionUID = 2340131505540279792L;
	
	/**
	 * An empty {@link NoHiddenWeightException exception}.
	 */
	public NoHiddenWeightException() {
		super();
	}
}
