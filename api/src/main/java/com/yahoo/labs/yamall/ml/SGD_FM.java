package com.yahoo.labs.yamall.ml;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Random;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.core.SparseVector;


/**
 * Stochastic Gradient Descent for Two way Factorization Machines.
 * 
 * 
 * <p>
 * The details of the algorithm are from 1) Steffen Rendle, "Factorization Machines" ,  
 * 2) S. Ross, P. Mineiro, J. Langford, "Normalized online learning", UAI 2013.								 
 * <p>
 * 
 * <p>
 *  https://github.com/srendle/libfm
 * <p>
 * 
 * Memory required - 2^bits * 6 * 8 bytes
 * 
 *  w - parameters of linear model
 *  v - parameters of interaction parameters(two way interaction)
 *  
 *   @author Krishna Chaitanya Chakka, Francesco Orabona
 *   @version 1.1
 * 
 */
@SuppressWarnings("serial")
public class SGD_FM implements Learner {
	
	private double eta = .5;
	private double epsilon = 1e-6;
	private Loss lossFnc;
    private int size_hash = 0;
    private int fmNumberFactors = 0;
    private double[] w;             //linear model parameters
    private double[] s;
    
    private double[][] v;           //interaction parameters
    private double[] sumProd_v;     
    
    // past gradients information for adaptive update 
    private double[] gradientSquare_w;  
    private double[][] gradientSquare_v;
    
    
	public SGD_FM(int bits, int fmNumberFactors) {
		size_hash = 1 << bits;
		this.fmNumberFactors = fmNumberFactors;
		w = new double[size_hash];
		s = new double[size_hash];
		
		//TODO: optimize space
		v = new double[size_hash][fmNumberFactors];  
		
		// initialize all weights with Gaussian noise
		init(size_hash, fmNumberFactors);  
		sumProd_v = new double[fmNumberFactors];
		gradientSquare_w = new double[size_hash];
		gradientSquare_v = new double[size_hash][fmNumberFactors];	
	}
	
	private void init(int hash_size, int numFactors) {
		/*
		 * Initialize the interaction parameters with Gaussian noise
		 *        to avoid the gradients to be 0
		 */
		Random r = new Random();
		for (int i = 0 ; i < hash_size; i++)
			for (int j = 0; j < numFactors; j++)  
				v[i][j] = r.nextGaussian() * 0.01; // mean = 0, variance = 0.01
	}
	
	public double update(Instance sample) {
		/*
		 *  calculate pred => sum(w_i*x_i)
		 */
		double pred = predict_normalized_features(sample);		
		final double negativeGrad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());
		
		// Update linear weights
		for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
			int key = entry.getIntKey();	
			double x_i = entry.getDoubleValue();
			double w_i = w[key];
			double tmp = negativeGrad * x_i;
			gradientSquare_w[key] += (tmp * tmp);
			double eta_grad = eta / Math.sqrt(gradientSquare_w[key] + epsilon);
			w[key] = w_i + eta_grad * negativeGrad * x_i;
		}
		// Update interaction weights
		for (int i = 0; i < fmNumberFactors; i++) {
			for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
				int key = entry.getIntKey();
				double x_i = entry.getDoubleValue();
				double v_ij = v[key][i];
				double v_grad = x_i*sumProd_v[i] - v_ij * x_i * x_i;
				double tmp = negativeGrad * v_grad;
				gradientSquare_v[key][i] += (tmp * tmp);
				double eta_grad = eta / Math.sqrt(gradientSquare_v[key][i] + epsilon);
				v[key][i] = v_ij + eta_grad * negativeGrad * v_grad;
			}
		}
		return pred;
	}

	private double predict_normalized_features(Instance sample) {
		double pred = 0;
		
		//one-way interaction
		for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
			int key = entry.getIntKey();
			double s_i = s[key];
			double x_i = entry.getDoubleValue();
			double w_i = w[key];
			if (Math.abs(x_i) > s_i) {
				double tmp = s_i / Math.abs(x_i);
				s[key] = Math.abs(x_i);
				if (tmp!=0) {
					for (int k = 0 ; k < fmNumberFactors; k++)
						v[key][k] *= tmp;
	                w_i *= tmp;
	                w[key] = w_i;
				}                
            }
			pred += (x_i * w_i);
		}
		
        // Calculating two way interaction: O(nk)
		for (int i = 0; i < fmNumberFactors; i++) {
			double linearSum = 0;
			double squareSum = 0;
			for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
				int key = entry.getIntKey();
				double x_i = entry.getDoubleValue();
				
				double prod = v[key][i] * x_i;
				linearSum += prod;
				squareSum += prod * prod;
			}
			sumProd_v[i] = linearSum;
			pred += 0.5*(linearSum*linearSum - squareSum);
		}
		
		return pred;
	}
	
	
	public double predict(Instance sample) {
		double pred = 0;
		
		//one-way interaction
		for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
			int key = entry.getIntKey();
			pred += (entry.getDoubleValue() * w[key]);
		}
		
		// Calculating two way interaction: O(nk)
		for (int i = 0; i < fmNumberFactors; i++) {
			double linearSum = 0;
			double squareSum = 0;
			for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
				int key = entry.getIntKey();
				double prod = v[key][i] * entry.getDoubleValue();
				linearSum += prod;
				squareSum += prod * prod;
			}
			pred += 0.5*(linearSum * linearSum - squareSum);
		}
		
		return pred;
	}

	public String toString() {
        String tmp = "Using Factorization Machines optimizer (adaptive and normalized)\n";
        tmp = tmp + "Number of factors = " + fmNumberFactors + "\n";
        tmp = tmp + "Initial learning rate = " + eta + "\n";
        tmp = tmp + "Loss function = " + getLoss().toString();
        return tmp;
    }
	
	public void setLoss(Loss lossFnc) {
		this.lossFnc = lossFnc;
	}

	public Loss getLoss() {
		return lossFnc;
	}

	public void setLearningRate(double eta) {
		this.eta = eta;
	}

	public SparseVector getWeights() {
		// TODO Auto-generated method stub
		return null;
	}
	
	private void writeObject(ObjectOutputStream o) throws IOException {
        o.defaultWriteObject();
    }
}
