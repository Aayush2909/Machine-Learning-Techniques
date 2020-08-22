import numpy as np
import random

class Perceptron:
    
    def __init__(self, n_inputs, n_hidden, n_outputs):
        
        layers = [n_inputs] + n_hidden + [n_outputs]   #total no. of layers
        
        #initializing some random weight edges
        self.weights = []
        for i in range(len(layers)-1):
            
            weight = np.random.rand(layers[i], layers[i+1])
            self.weights.append(weight)
        
        #For storing activation/ final output of a layer for backward pass
        activations = []
        for i in range(len(layers)):
            act = np.zeros(layers[i])
            activations.append(act)
        self.activations = activations
        
        #For storing derivatives(dE/dW[i]) during backward pass
        der = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            der.append(act)
        self.derivatives = der
            
            
    #forward pass in a perceptron    
    def forward_propagate(self, inputs):
        
        activations = inputs
        self.activations[0] = activations
        for count,i in enumerate(self.weights):
            
            output = np.dot(activations, i)
            activations = self._sigmoid(output)
            self.activations[count+1] = activations   #activations of each layer stored
            
        return activations  #output of the perceptron
    
    
    #back-pass
    def backward_propagate(self, error):
        
        for i in reversed(range(len(self.derivatives))):
            
            act = self.activations[i+1]                                         # a[i+1]                    
            delta = error*self._sigmoid_differentiate(act)                      # calculating error*f'(a[i+1]) --> error*a[i+1]*(1-a[i+1])         
            delta_reshaped = delta.reshape(delta.shape[0], -1).T                # reshaping for dot product             
            curr_act = self.activations[i]                                      # a[i]         
            curr_act_reshaped = curr_act.reshape(curr_act.shape[0], -1)         # reshaping for dot product      
            self.derivatives[i] = np.dot(curr_act_reshaped, delta_reshaped)     # storing derivatives         
            error = np.dot(delta, self.weights[i].T)                            # error --> error*f'(a[i+1])*W[i] for next back layer because dE/dW = error*f'(a[i+1])*W[i]*f'(a[i])*a[i-1]   
            
    
    
    #calculating gradient descent
    def grad_descent(self, learning_rate):
        for i in range(len(self.weights)):
            w = self.weights[i]
            d = self.derivatives[i]
            w += d*learning_rate
            self.weights[i] = w
            
            
    
    #training the perceptron
    def training(self, inputs, targets, epoch=500, learning_rate=0.2):
        
        print("Training starts!!!")
        for i in range(epoch):
            mse = 0
            for inp, tar in zip(inputs, targets):
                
                out = self.forward_propagate(inp)
                error = tar - out
                mse += error**2
                self.backward_propagate(error)
                self.grad_descent(learning_rate)
                
            print("Epoch No.{}, Error: {}".format(i+1, mse/len(targets)))
    
        print("Perceptron Trained!!!")
        print("##################################")
    
    
    
    #sigmoidal activation function
    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    #sigmoidal differentiation
    def _sigmoid_differentiate(self, x):
        return x*(1.0-x)
    
if __name__ == '__main__':
    
    p = Perceptron(2,[3,5,3],1)
    
    inputs = np.array([[random.random()/2 for _ in range(2)] for _ in range(1000)])     #input dataset for traing
    targets = np.array([i[0]+i[1] for i in inputs])    #target dataset for error calculation/ training the perceptron
    
    p.training(inputs, targets,epoch=500, learning_rate=0.5)   #perceptron training
    
    print()
    print("A multilayer perceptron to perform Addition-")
    
    test_input = []   #testing data
    x = float(input("Enter first number: "))
    test_input.append(x)
    y = float(input("Enter second number: "))
    test_input.append(y)
    
    result = p.forward_propagate(test_input)   #perceptron result on test data
    
    print("The perceptron determines the sum of input {} and {} to be: {}".format(test_input[0], test_input[1], result))
    
