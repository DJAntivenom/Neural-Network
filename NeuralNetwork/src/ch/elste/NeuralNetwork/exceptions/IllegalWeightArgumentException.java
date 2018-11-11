package ch.elste.NeuralNetwork.exceptions;

public class IllegalWeightArgumentException extends IllegalArgumentException {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6798952518319069995L;

	/**
	 * Thrown if an illegal weight argument is given.
	 */
	public IllegalWeightArgumentException() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * Thrown if an illegal weight argument is given.
	 * 
	 * @param s
	 */
	public IllegalWeightArgumentException(String s) {
		super(s);
		// TODO Auto-generated constructor stub
	}

	/**
	 * Thrown if an illegal weight argument is given.
	 * 
	 * @param cause
	 */
	public IllegalWeightArgumentException(Throwable cause) {
		super(cause);
		// TODO Auto-generated constructor stub
	}

	/**
	 * Thrown if an illegal weight argument is given.
	 * 
	 * @param cause
	 * @param s
	 */
	public IllegalWeightArgumentException(String message, Throwable cause) {
		super(message, cause);
		// TODO Auto-generated constructor stub
	}

}
