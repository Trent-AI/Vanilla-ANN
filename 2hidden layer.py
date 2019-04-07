import numpy as np
import random as r
import matplotlib.pyplot as plt
r.seed(1)

class NeuralNet:
    def __init__(self, starting, num1Neurons, num2Neurons,output):
        #Initialize the Weights in Each Layer with Random Values
        self.syn1 = np.random.random((starting,num1Neurons))*2-1
        self.syn2 = np.random.random((num1Neurons,num2Neurons))*2-1
        self.syn3 = np.random.random((num2Neurons,output))*2-1
        self.err = 0

    #Signomd Acivation funtion to add non-linearity
    def sigmoid(self, x):
        x = 1 / (np.exp(-x) + 1)
        return x

    #Sigmoid derivative for backpropagation
    def sigmoid_dev(self, x):
       return x * ( 1 - (x))

    #Relu activation function for non-linearity
    def relu(self,x):
        x[x<=0] = 0
        x[x>0] = x
        return x

    #Relu derivative for backpropagation
    def relu_dev(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    #Forward passing functions for relu and sigmoid
    def sigThink(self,inputs):
        self.layer1 = self.sigmoid(np.dot(inputs,self.syn1))
        self.layer2 = self.sigmoid(np.dot(self.layer1,self.syn2))
        self.layer3 = self.sigmoid(np.dot(self.layer2,self.syn3))
        
    def reluThink(self,inputs):
        self.layer1 = self.relu(np.dot(inputs,self.syn1))
        self.layer2 = self.relu(np.dot(self.layer1,self.syn2))
        self.layer3 = self.relu(np.dot(self.layer2,self.syn3))

    #Output function to test network
    def sigGuess(self, inputs):
        self.sigThink(inputs)
        return self.layer3
    def reluGuess(self, inputs):
        self.reluThink(inputs)
        return self.layer3
    
    
    def sigTrain(self,inputs,answer,iters):
        for its in range(iters):
            #Do a forward pass
            self.sigThink(inputs)
            #Find MSE error derivative
            error3 = answer - self.layer3
            #Chain Rule for sigmoid activation
            delta3 = error3*self.sigmoid_dev(self.layer3)
            #Backwards pass error to previos layer and repeat
            error2 = np.dot(delta3,self.syn3.T)
            delta2 = error2*self.sigmoid_dev(self.layer2)
            error1 = np.dot(delta2,self.syn2.T)
            delta1 = error1*self.sigmoid_dev(self.layer1)
            #multiply by activations in final chain rule peice
            self.syn1 += np.dot(inputs.T,delta1)
            self.syn2 += np.dot(self.layer1.T,delta2)
            self.syn3 += np.dot(self.layer2.T,delta3)
            self.err = 0
            #find average error in network in each run
            for i in range(error3.shape[0]):
                for s in range(error3.shape[1]):
                    self.err += abs(error3[i][s])
            self.err /= error3.shape[0]*error3.shape[1]
            #print("Iter",its,"of",iters,"with error",self.err)
        
    def reluTrain(self,inputs,answer,iters):
        for its in range(iters):
            self.reluThink(inputs)
            error3 = answer - self.layer3
            delta3 = error3*self.relu_dev(self.layer3)
            error2 = np.dot(delta3,self.syn3.T)
            delta2 = error2*self.relu_dev(self.layer2)
            error1 = np.dot(delta2,self.syn2.T)
            delta1 = error1*self.relu_dev(self.layer1)
            self.syn1 += np.dot(inputs.T,delta1)
            self.syn2 += np.dot(self.layer1.T,delta2)
            self.syn3 += np.dot(self.layer2.T,delta3)
            self.err = 0
            for i in range(error3.shape[0]):
                for s in range(error3.shape[1]):
                    self.err += abs(error3[i][s])
            self.err /= error3.shape[0]*error3.shape[1]
            print("Iter",its,"of",iters,"with error",self.err)
            
training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1],[1,0,0]])
training_set_outputs = np.array([[0, 0, 1, 1,1]]).T
net=NeuralNet(3,3,3,1)
net.sigTrain(training_set_inputs, training_set_outputs, 10000)
print("[0,1,0] ->")
print(net.sigGuess([[0,1,0]]))
      
                            


