import numpy as np

import json,gzip,pickle,urllib.request

def DeltaSum(a,b):

    output = 0

    for i in range(len(b)):
        output += (a*b[i])

    return output

def ElementMult(a, b):
    assert(len(a) == len(b))

    output = []
    
    for i in range(len(b)):
        output.append(a[i]*b[i])

    return output

def GradientDescent(weight,prediction,inp):
    deltaWeight = np.zeros((len(inp),len(delta)))
    for i in range(len(inp)):
        for j in range(len(delta)):
            deltaWeight[i] = np.dot(delta,inp[i])
    
    for i in range(len(weight)):
        for j in range(len(weight[i])):
            weight[i][j] -= alpha * deltaWeight[i][j]

    return weight

def NeuralNetwork(input,weight):

    prediction = np.dot(input,weight)

    return prediction


#Loading MNIST library
urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
with gzip.open('mnist.pkl.gz','rb') as f:
    trainSet, validSet, testSet = pickle.load(f,encoding='latin1')



alpha = 0.01
weights = np.zeros((784,10))
delta = np.zeros(10)
error = np.zeros(10)
goal = trainSet[1]
numbers = [[0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]]

for num in range(100):
    inp = trainSet[0][num]
    prediction = NeuralNetwork(inp,weights)
    currentGoal = goal[num]
    
    for i in range(len(prediction)):
        error[i] = (prediction[i]-numbers[currentGoal][i])**2
        delta[i] = prediction[i] - numbers[currentGoal][i]

    #print("Error: " + str(error) + " Prediction: " + str(prediction))

    weights = GradientDescent(weights,prediction,inp)

print("Final Prediction: " + str(prediction))
print("Final Error: " + str(error))