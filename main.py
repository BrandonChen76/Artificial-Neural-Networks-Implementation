import numpy as np
import pandas as pd

#Global stuff

#Learning rate
n = 0.3

#Number of neurons
neurons = 4

#Number of epoch
epoch = 100000

#Random weights (0,1) -.5 / 5 = (-0.1,0.1)
W1 = (np.random.rand(3, neurons) - 0.5) * .2
W2 = (np.random.rand(neurons, 1) - 0.5) * .2

#Training data
training_data = pd.DataFrame({
    'x1':[0,0,0,0,1,1,1,1],
    'x2':[0,0,1,1,0,0,1,1],
    'x3':[0,1,0,1,0,1,0,1],
    'c':[0,1,1,0,1,0,0,1]
})

#===========================================================================================================================================================================================

#All logic for the forward and backward propagation
#Input: Data frame of the training data, Nx(1-3) array of weight layer 1(W1), (1-3)x1 array of weight for layer 2(W2), number of hidden nerons to use(1-3), exepcted output
#Output: Returns weights, MSE, results
def forward_and_backward_propagation(input, weight1, weight2, hidden, expected):
    # print("OldW1")
    # print(weight1)
    # print("OldW2")
    # print(weight2)

    #Forward propagation
    #Get H
    H = np.dot(weight1.T, input)
    H = 1 / (1 + np.exp(-1 * H))
    # print("H")
    # print(H)

    #Get Y
    Y = np.dot(weight2.T, H)
    Y = 1 / (1 + np.exp(-1 * Y))
    # print("Y")
    # print(Y)

    error = expected - Y
    # print("Error:", error)

    #Backward propagation

    #Get S1
    S1 = Y * (1 - Y) * (expected - Y)
    # print("S1")
    # print(S1)

    #Get S2
    S2 = H * (1 - H) * np.dot(weight2, S1)
    # print("S2")
    # print(S2)

    #Update weights

    #W1
    weight1 = weight1 + n * np.outer(input, S2.T)
    # print("NewW1")
    # print(weight1)

    #W2
    weight2 = weight2 + (n * H * S1)
    # print("NewW2")
    # print(weight2)

    return weight1, weight2, error, Y

#===============================================================================================================================================================================================

for i in range(0, epoch, 1):

    sqr_error_total = 0

    correct = 0

    #each epoch
    for j, row in training_data.iterrows():

        #Get expected
        T = row['c']

        #Convert input to Nx1 array
        X = row[['x1', 'x2', 'x3']].to_numpy().reshape(-1, 1)

        #Start the algorithm  and update the weights
        W1, W2, error, Y = forward_and_backward_propagation(X, W1, W2, neurons, T)

        #Error for each epoch
        sqr_error_total += np.square(error)

        if(i == epoch - 1 or i == 0):
            predicted = 1 if Y >= 0.5 else 0
            if(predicted == T):
                correct += 1

    if(i == epoch - 1 or i == 0):
        print(correct/8)
        print(sqr_error_total/8)
