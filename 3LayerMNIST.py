import sys,numpy as np 
from keras.datasets import mnist
#Using Dropuout Regularization and batch gradient descent
#tahn and softmax as activation functions


(xTest,yTest), (xTrain,yTrain) = mnist.load_data()

images, labels = (xTrain[0:1000].reshape(1000,28*28)/255, yTrain[0:1000])

oneHotLabels = np.zeros((len(labels),10))

for i,l in enumerate(labels):
    oneHotLabels[i][l] = 1
labels = oneHotLabels

testImages = xTest.reshape(len(xTest),28*28)/255
testLabels = np.zeros((len(yTest),10))
for i,l in enumerate(yTest):
    testLabels[i][l] = 1

def tanh(x):
    return np.tanh(x)
def tanh2deriv(x):
    return 1 -(x**2)

def softmax(x):
    temp = np.exp(x)
    return temp/ np.sum(temp,axis=1,keepdims=True)

np.random.seed(1)

alpha, iterations, hiddenSize = (2,300,100)
pixelPerImage, numLabels = (784,10)
batchSize = 100

weights01=0.02*np.random.random((pixelPerImage,hiddenSize))-0.01
weights12 = 0.2*np.random.random((hiddenSize,numLabels))-0.1

for j in range(iterations):
    correctCnt = 0
    for i in range(int(len(images)/batchSize)):
        batchStart,batchEnd = ((i*batchSize),((i+1)*batchSize))
        layer0 = images[batchStart:batchEnd]
        layer1 = tanh(np.dot(layer0,weights01))
        dropoutMask = np.random.randint(2,size=layer1.shape)
        layer1 *= dropoutMask*2
        layer2 = softmax(np.dot(layer1,weights12))

        for k in range(batchSize):#Batch Gradient Descent
            correctCnt += int(np.argmax(layer2[k:k+1])==np.argmax(labels[batchStart+k:batchEnd+k+1]))

        layer2Delt = (labels[batchStart:batchEnd]-layer2)/(batchSize*layer2.shape[0])
        layer1Delt = layer2Delt.dot(weights12.T)*tanh2deriv(layer1)

        layer1Delt *= dropoutMask

        weights12 += alpha*layer1.T.dot(layer2Delt)
        weights01 += alpha*layer0.T.dot(layer1Delt)



    testCorrectCnt = 0
    for it in range(len(testImages)):
        layer0 = testImages[it:it+1]
        layer1 = tanh(np.dot(layer0,weights01))
        layer2 = np.dot(layer1,weights12)
        testCorrectCnt += int(np.argmax(layer2)==np.argmax(testLabels[it:it+1]))

    if(j%10 == 0):
        sys.stdout.write("\n"+"I:"+str(j)+"Test-Accuracy:"+str(testCorrectCnt/float(len(testImages)))+"Train-Accuracy:" + str(correctCnt/float(len(images))))