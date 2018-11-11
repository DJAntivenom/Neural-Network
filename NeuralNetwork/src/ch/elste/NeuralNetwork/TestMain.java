package ch.elste.NeuralNetwork;

import java.util.Arrays;

@Deprecated
public class TestMain {

	public static void main(String[] args) {
		SupervisedNeuralNetwork nn = new SupervisedNeuralNetwork(.5, 2, 1, 4);
		double data[][] = {{0d,1d},{1d},{1d,0d},{1d},{0d,0d},{0d},{1d,1d},{0d}};
		//nn.print();
		for(int i = 0; i < 20000; i++) {
			int j = Math.round((float) Math.random()*3)*2;
			nn.backpropagate(data[j], data[j+1]);
		}
		
		System.out.println(Arrays.toString(nn.feedForward(data[0])));
		System.out.println(Arrays.toString(nn.feedForward(data[2])));
		System.out.println(Arrays.toString(nn.feedForward(data[4])));
		System.out.println(Arrays.toString(nn.feedForward(data[6])));
		
		//nn.print();
	}

}
