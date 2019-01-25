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

class LogisticRegression:

    def GetNNInput(self, img):

        img_flatten = img.reshape(img.shape[0], -1).T
        img_flatten = img_flatten/255.

        return img_flatten

    def InitializeParams(self, nn_in_size, nn_out_size):
        w = np.random.randn(nn_out_size, nn_in_size)*0.01
        b = np.zeros((nn_out_size, 1))

        return w, b

    def PropagateForward(self, input, w, b):

        #linear output
        z = np.matmul(w, input) + b

        #non - linear output
        out = self.Sigmoid(z)

        return out 

    def ComputeCost(self, nn_out, true_result, nn_train_size):

        cost = -(1/nn_train_size)*np.sum(true_result*np.log(nn_out) + (1 - true_result)*np.log(1 - nn_out))
        return cost

    def PropagateBackward(self, nn_in, nn_out, true_result, nn_train_size):

        dw = (1/nn_train_size)*np.matmul(nn_in, (nn_out - true_result).T).T
        db = (1/nn_train_size)*np.sum(nn_out - true_result, axis = 1, keepdims = True)
        db = np.squeeze(db)

        return dw, db

    def Sigmoid(self, input):
        out = 1/(1 + np.exp(-input))
        return out

    def Optimize(self, nn_in, y_train, learning_rate, num_of_iteration, nn_in_size, nn_train_size, nn_out_size):

        w, b = self.InitializeParams(nn_in.shape[0], nn_out_size)
        if __debug__:
            assert(w.shape == (nn_out_size, nn_in_size))
            assert(b.shape == (nn_out_size, 1))

        for i in range(num_of_iteration):
            nn_out = self.PropagateForward(nn_in, w, b)

            if __debug__:
                assert(nn_out.shape == (1, nn_train_size))

            cost = self.ComputeCost(nn_out, y_train, nn_train_size)

            dw, db = self.PropagateBackward(nn_in, nn_out, y_train, nn_train_size)
            
            if __debug__:
                assert(dw.shape == w.shape)
                assert(db.dtype == float)

            w = w - learning_rate*dw
            b = b - learning_rate*db 

            if (i % 10 == 0):
                print("iteration " + str(i) + ": " + "cost: ", cost)

        return w, b, cost

    def Predict(self, nn_in, w, b, sample_size, binary_threshold):

        z = np.matmul(w, nn_in) + b
        nn_out = self.Sigmoid(z)

        prediction = np.zeros((1, sample_size))

        for i in range(sample_size):
            if (nn_out[0, i] > binary_threshold):
                prediction[0, i] = 1
            else:
                prediction[0, i] = 0
                
        return prediction

def main():
    
    #hyper parameter setting
    nn_in_size = 64*64*3
    nn_out_size = 1
    nn_train_size = 19
    nn_test_size = 19
    num_of_iteration = 500
    learning_rate = 0.009
    img_rows = 64
    img_cols = 64
    img_channels = 3
    binary_threshold = 0.5

    lrg = LogisticRegression()

    #load training and testing output   
    y_train = LoadTxtData(train_path, "y_train")
    y_test = LoadTxtData(test_path, "y_test")

    #load training image and resize to be 64, 64, 3
    standard_img = np.zeros((nn_train_size, img_rows, img_cols, img_channels))
    for i in range(1, 20):
        file_path = train_path + str(i) + ".jpg"
        raw_img = cv2.imread(file_path)
        standard_img[i - 1,:,:,:] = ResizeImg(raw_img, img_rows, img_cols)
    
    #start training
    nn_train_in = lrg.GetNNInput(standard_img)        
    if __debug__:
        assert(nn_train_in.shape == (nn_in_size, nn_train_size))
    
    w, b, cost = lrg.Optimize(nn_train_in, y_train, learning_rate, num_of_iteration, nn_in_size, nn_train_size, nn_out_size)

    #check training accuracy
    nn_train_predict = lrg.Predict(nn_train_in, w, b, nn_train_size, binary_threshold)
    print("train prediction: ", nn_train_predict)
    print("train ground truth: ", y_train)
    print("accuracy: ", ComputeAccuracy(nn_train_predict, y_train))

    #load testing image and resize to be 64, 64, 3
    standard_img = np.zeros((nn_test_size, img_rows, img_cols, img_channels))
    for i in range(1, 20):
        file_path = test_path + str(i) + ".jpg"
        raw_img = cv2.imread(file_path)
        standard_img[i - 1,:,:,:] = ResizeImg(raw_img, img_rows, img_cols)
    
    nn_test_in = lrg.GetNNInput(standard_img)        
    if __debug__:
        assert(nn_test_in.shape == (nn_in_size, nn_test_size))
    
    #check testing accuracy
    nn_test_predict = lrg.Predict(nn_test_in, w, b, nn_test_size, binary_threshold)
    print("test prediction: ", nn_test_predict)
    print("test ground truth: ", y_test)
    print("accuracy: ", ComputeAccuracy(nn_test_predict, y_test))

if __name__ == '__main__':
    main()