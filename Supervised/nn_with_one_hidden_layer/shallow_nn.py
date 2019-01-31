import numpy as np
import cv2

train_path = "D:/github/MachineLearning/Supervised/training_set/"
test_path = "D:/github/MachineLearning/Supervised/testing_set/"

def LoadTxtData(path, file_name):
    data = np.loadtxt(path + file_name + ".txt")
    return data

def ResizeImg(img, row_size, col_size):
    img = cv2.resize(img, (row_size, col_size))
    return img

def ComputeAccuracy(est_out, true_out):
    return (100 - np.mean(np.abs(true_out - est_out))*100)

class ShallowNeuralNetwork:
    def GetNNInput(self, img):

        img_flatten = img.reshape(img.shape[0], -1).T
        img_flatten = img_flatten/255.

        return img_flatten

    def InitializeParams(self, dims_of_layers):
        L = len(dims_of_layers)

        parameters = {}

        for i in range(1, L):
            parameters['W' + str(i)] = np.random.randn(dims_of_layers[i], dims_of_layers[i - 1])*0.01
            parameters['b' + str(i)] = np.zeros((dims_of_layers[i], 1))

            if __debug__:
                assert(parameters['W' + str(i)].shape == (dims_of_layers[i], dims_of_layers[i - 1]))
                assert(parameters['b' + str(i)].shape == (dims_of_layers[i], 1))

        return parameters

    def ComputeCost(self, nn_out, true_result, nn_train_size):

        cost = (1/nn_train_size)*np.sum(-true_result*np.log(nn_out) - (1 - true_result)*np.log(1 - nn_out))
       
        if __debug__:
            assert(isinstance(cost, float))

        return cost

    def PropagateForward(self, input, parameters):

        W1 = parameters['W1']
        W2 = parameters['W2']
        b1 = parameters['b1']
        b2 = parameters['b2']

        Z1 = np.matmul(W1, input) + b1
        A1 = self.RELU(Z1)
        Z2 = np.matmul(W2, A1) + b2
        A2 = self.Sigmoid(Z2)

        if __debug__:
            assert(A2.shape == (1, input.shape[1]))

        caches = {"Z1": Z1,
                  "A1": A1,
                  "Z2": Z2,
                  "A2": A2}

        return A2, caches

    def PropagateBackward(self, nn_in, nn_out, true_result, parameters, caches, nn_train_size):
        #specify for one hidden layer

        W1 = parameters['W1']
        W2 = parameters['W2']

        Z1 = caches['Z1']
        Z2 = caches['Z2']
        A1 = caches['A1']
        A2 = caches['A2']
        
        dZ2 = nn_out - true_result
        dW2 = (1/nn_train_size)*np.matmul(dZ2, A1.T)
        db2 = (1/nn_train_size)*np.sum(dZ2, axis = 1, keepdims = True)
        db2 = np.squeeze(db2)

        dA1 = A1
        dA1[dA1 >= 0] = 1
        dA1[dA1 < 0] = 0
        dZ1 = np.matmul(W2.T, dZ2)*dA1
        dW1 = (1/nn_train_size)*np.matmul(dZ1, nn_in.T)
        db1 = (1/nn_train_size)*np.sum(dZ1, axis = 1, keepdims = True)

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads

    def Optimize(self, nn_in, y_train, learning_rate, num_of_iteration, nn_train_size, dims_of_layers):

        parameters = self.InitializeParams(dims_of_layers)

        for i in range(num_of_iteration):

            nn_train_out, caches = self.PropagateForward(nn_in, parameters)

            cost = self.ComputeCost(nn_train_out, y_train, nn_train_size)

            grads = self.PropagateBackward(nn_in, nn_train_out, y_train, parameters, caches, nn_train_size)

            parameters['W1'] = parameters['W1'] - learning_rate*grads['dW1']
            parameters['W2'] = parameters['W2'] - learning_rate*grads['dW2']
            parameters['b1'] = parameters['b1'] - learning_rate*grads['db1']
            parameters['b2'] = parameters['b2'] - learning_rate*grads['db2']

            if (i % 100 == 0):
                print("iteration " + str(i) + ": " + "cost: ", cost)

        return parameters   
        
    def Predict(self, nn_in, parameters, sample_size, binary_threshold):

        nn_out, cache = self.PropagateForward(nn_in, parameters)

        prediction = np.zeros((1, sample_size))

        for i in range(sample_size):
            if (nn_out[0, i] > binary_threshold):
                prediction[0, i] = 1
            else:
                prediction[0, i] = 0
         
        prediction = np.squeeze(prediction) 
                       
        return prediction

    def RELU(self, input):
        out = input
        out[out < 0] = 0
        return out

    def Sigmoid(self, input):
        out = 1/(1 + np.exp(-input))
        return out

def main():
    #hyper parameter setting
    img_rows = 100
    img_cols = 100
    img_channels = 3
    nn_in_size = img_rows*img_cols*img_channels
    nn_out_size = 1
    nn_train_size = 22
    nn_test_size = 19
    num_of_iteration = 2500
    learning_rate = 0.009
    dims_of_layers = [nn_in_size, 10, 1]
    binary_threshold = 0.5

    #load training and testing output   
    y_train = LoadTxtData(train_path, "y_train")
    y_test = LoadTxtData(test_path, "y_test")

    #load training image and resize to be 64, 64, 3
    standard_img = np.zeros((nn_train_size, img_rows, img_cols, img_channels))
    for i in range(1, nn_train_size + 1):
        file_path = train_path + str(i) + ".jpg"
        raw_img = cv2.imread(file_path)
        standard_img[i - 1,:,:,:] = ResizeImg(raw_img, img_rows, img_cols)

    
    #start training
    snn = ShallowNeuralNetwork()

    nn_train_in = snn.GetNNInput(standard_img)        
    if __debug__:
        assert(nn_train_in.shape == (nn_in_size, nn_train_size))

    parameters = snn.Optimize(nn_train_in, y_train, learning_rate, num_of_iteration, nn_train_size, dims_of_layers)

    nn_train_prediction = snn.Predict(nn_train_in, parameters, nn_train_size, 0.5)
    print("accuracy: ", ComputeAccuracy(nn_train_prediction, y_train))

    #start testing
    standard_img = np.zeros((nn_test_size, img_rows, img_cols, img_channels))
    for i in range(1, nn_test_size + 1):
        file_path = test_path + str(i) + ".jpg"
        raw_img = cv2.imread(file_path)
        standard_img[i - 1,:,:,:] = ResizeImg(raw_img, img_rows, img_cols)

    nn_test_in = snn.GetNNInput(standard_img)        
    if __debug__:
        assert(nn_test_in.shape == (nn_in_size, nn_test_size))
    nn_test_prediction = snn.Predict(nn_test_in, parameters, nn_test_size, 0.5)

    print("resut1: ", nn_train_prediction)
    print("resut2: ", nn_test_prediction)
    print("resut3: ", y_test)
    print("accuracy: ", ComputeAccuracy(nn_test_prediction, y_test))


if __name__ == '__main__':
    main()