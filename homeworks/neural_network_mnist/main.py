# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.F1: torch.FloatTensor = None
        #self.sigma: torch.nn.functional = None #our sigma non-linear function will be the ReLU

        #initialzing h,z,d
        self.h = h #h = 64
        self.d = d #for MNIST d = 784
        self.k = k #for MNIST k = 10

        #initializing W_0 and b_0 parameters according to alpha formula and sampling from uniform distribution where the bounds are h and d
        alpha1 = 1/math.sqrt(self.d)
        u1 = Uniform(-alpha1, alpha1)
        self.weight0 = Parameter(u1.sample([self.h, self.d]).float())
        self.bias0 = Parameter(u1.sample((self.h,)).float().float())

        #initializing W_1 and b_1 parameters according to alpha formula and sampling from uniform distribution where the bounds are k and h
        alpha2 = 1/math.sqrt(self.h)
        u2 = Uniform(-alpha2, alpha2)
        self.weight1 = Parameter(u2.sample([self.k, self.h]).float())
        self.bias1 = Parameter(u2.sample((self.k,)).float())

        return None
        raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        # we know W is defined generally to be \in R^(n x m) and b \in R^n
        # alpha = 1/√m and we must initialize 
        # for F1, we know W_0 \in R^(h x d), b_0 \in R^h, W_1 \in R^(k x h), and b_1 \in R^k
        self.x = x #setting self.X to x now

        #performing first linear regression layer
        layer1 = self.x @ self.weight0.T + self.bias0

        #performing relu after the first hidden layer
        sigma = relu(layer1)
        
        #computing F1's output with linear regression at the end
        self.F1 = sigma @ self.weight1.T + self.bias1
        return self.F1
        raise NotImplementedError("Your Code Goes Here")


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()

        self.F2: torch.FloatTensor = None
        #self.sigma1: torch.nn.functional = None #our sigma non-linear function will be the ReLU
        #self.sigma2: torch.nn.functional = None 

        self.h0 = h0 #h0 = 32
        self.h1 = h1 #h1 = 32
        self.d = d #for MNIST d = 784
        self.k = k #for MNIST k = 10

        #initializing W_0 and b_0 parameters according to alpha formula and sampling from uniform distribution where the bounds are h0 and d
        alpha1 = 1/math.sqrt(self.d)
        u1 = Uniform(-alpha1,alpha1)
        self.weight0 = Parameter(u1.sample([self.h0, self.d]).float())
        self.bias0 = Parameter(u1.sample((self.h0,)).float())

        #initializing W_1 and b_1 parameters according to alpha formula and sampling from uniform distribution where the bounds are h1 and h0
        alpha2 = 1/math.sqrt(self.h0)
        u2 = Uniform(-alpha2,alpha2)
        self.weight1 = Parameter(u2.sample([self.h1, self.h0]).float())
        self.bias1 = Parameter(u2.sample((self.h1,)).float())

        #initializing W_2 and b_2 parameters according to alpha formula and sampling from uniform distribution where the bounds are k and h1
        alpha3 = 1/math.sqrt(self.h1)
        u2 = Uniform(-alpha3,alpha3)
        self.weight2 = Parameter(u2.sample([self.k, self.h1]).float())
        self.bias2 = Parameter(u2.sample((self.k,)).float())

        return None
        raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
         # we know W is defined generally to be \in R^(n x m) and b \in R^n
        # alpha = 1/√m and we must initialize 
        # for F1, we know W_0 \in R^(h0 x d), b_0 \in R^h0, W_1 \in R^(h1 x h0), b_1 \in R^h1, W_2 \in R^(k x h1), and b_2 \in R^k
        self.x = x #setting self.X to x now

        #performing first linear regression layer
        layer1 = self.x @ self.weight0.T + self.bias0

        #performing relu after the first hidden layer
        sigma1 = relu(layer1)
        
        #performing second linear regression layer
        layer2 = sigma1 @ self.weight1.T + self.bias1

        #performing relu after the second hidden layer
        sigma2 = relu(layer2)

        #computing F2's output with linear regression at the end
        self.F2 = sigma2 @ self.weight2.T + self.bias2
        return self.F2

        raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    #initialzing empty lists for losses and accuracy (not needed for latter)
    avg_losses = []
    accuracy_list = []

    #initializing 0s for epochs and accuracy
    epochs = 0
    accuracy = 0

    #we will use a while loop to iterate over until our model achieves 99% accuracy
    while accuracy < 0.99:
        epochs += 1 #incrementing number of epochs
        model.train() #setting model to train
        epoch_losses= [] #creating an empty list of losses in each batch per epoch
        trues = 0 #initialzing true predictions for model in each epoch
        total = 0 # #initialzing total data in each epoch
        for data, targets in train_loader:
            optimizer.zero_grad() #zero_grad resets the optimizer's gradient to zero before we begin to update it
            outputs = model(data) #computing outputs of our model
            loss = cross_entropy(outputs,targets) #computing loss of the model's output vs targets using cross entropy loss as specified for the problem
            epoch_losses.append(loss.item()) #appending loss in thus batch to the epoch's loss list
            loss.backward() #computing backward step for loss
            optimizer.step() #compute gradient descent 
            with torch.no_grad(): #now with torch.no_grad (refer to train.py file for A4 for more info)
                preds = outputs.argmax(dim=1) #obtaining the prediction with the highest probability
                trues += (preds == targets).sum().item() #incrementing the number of true predictions
                total += targets.size(0) #incrementing the total

        
        avg_loss = torch.mean(torch.FloatTensor(epoch_losses)) #computing average loss per epoch
        avg_losses.append(avg_loss) #appending this list to the list of losses for each and every epoch
        accuracy = trues/total #computing the accuracy at this epoch
        accuracy_list.append(accuracy) #appending to accuracy list (not needed)
    
    return avg_losses
    raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    train_set = TensorDataset(x,y)
    test_set = TensorDataset(x_test,y_test)

    train_loader = DataLoader(train_set, batch_size = 64, shuffle = True) #batch size = 64 because that's what the PyTorch NN notebook used (arbitrarily)
    test_loader = DataLoader(test_set, batch_size = 64, shuffle = True)

    #part (a)
    #we train with model F1
    #here we let h = 64 (for the number of hidden units) and we know d = 784 and k = 10
    h = 64
    d = 784
    k = 10

    #creating F1 model instance
    f1 = F1(h,d,k)
    opt1 = Adam(f1.parameters(), lr = 0.001) #creating adam optimizer for F1
    f1_losses = train(f1, opt1, train_loader) #training the F1 model and obtaining its average losses per epoch

    #plotting the average loss per epoch for F2
    plt.figure(1, figsize = (12,5))
    #plt.plot(torch.arange(1,len(f1_losses)+1), f1_losses)
    plt.plot(f1_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Cross Entropy Loss Training History for F1')

    #testing the F1 model on the test data
    f1.eval()
    with torch.no_grad(): 
        trues1 = 0 #initializing number of correct predictions for F1
        total1 = 0 #initializing total number of predictions for F1
        total_tloss1 = 0 #initializing total test loss for F1
        for data1, targets1 in test_loader: #iterating through batches of test dataloader
            test_output1 = f1(data1) #computing outputs with the model
            test_loss1 = cross_entropy(test_output1,targets1) #computing loss of the model with cross entropy
            total_tloss1 += test_loss1.item() #incrementing total loss using .item of tensor
            test_pred1 = test_output1.argmax(dim = 1) #obtaining predictions with argmax function of pytorch, source: https://docs.pytorch.org/docs/stable/generated/torch.argmax.html
            trues1 += (test_pred1 == targets1).sum().item() #incrementing number of correct predictions using .item of tensor
            total1 += targets1.size(0) #incrementing total predictions using .item of tensor
        
    accuracy1 = trues1/total1 #computing test accuracy
    print(f'F1 Model Test Accuracy: {accuracy1}')
    print(f'F1 Model Test Loss: {total_tloss1}')
    
    params1 = sum(param.numel() for param in f1.parameters()) #computing number of parameters using .numel from pytorch, source: https://docs.pytorch.org/docs/stable/generated/torch.numel.html
    print(f"F1 Model's Number of Parameters: {params1}")

    # part (b)
    h0 = 32
    h1 = 32
    d = 784
    k = 10

    f2 = F2(h0,h1,d,k) #creating F2 model instance
    opt2 = Adam(f2.parameters(), lr = 0.001) #creating adam optimizer for F2
    f2_losses = train(f2, opt2, train_loader) #training the F2 model and obtaining its average losses per epoch

    #plotting the average loss per epoch for F2
    plt.figure(2, figsize = (12,5))
    #plt.plot(torch.arange(1,len(f1_losses)+1), f1_losses)
    plt.plot(f2_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Cross Entropy Loss Training History for F2')

    #testing the F2 model on the test data
    f2.eval()
    with torch.no_grad(): 
        trues2 = 0 # initializing number of correct predictions for F2
        total2 = 0 #initializing total number of predictions for F2
        total_tloss2 = 0 #initializing total test loss for F2
        for data2, targets2 in test_loader: #iterating through batches of test dataloader
            test_output2 = f2(data2) #computing outputs with the model
            test_loss2 = cross_entropy(test_output2,targets2) #computing loss of the model with cross entropy
            total_tloss2 += test_loss2.item() #incrementing total loss using .item of tensor
            test_pred2 = test_output2.argmax(dim=1) #obtaining predictions with argmax function of pytorch, source: https://docs.pytorch.org/docs/stable/generated/torch.argmax.html
            trues2 += (test_pred2 == targets2).sum().item() #incrementing number of correct predictions using .item of tensor
            total2 += targets2.size(0) #incrementing total predictions using .item of tensor
        
    accuracy2 = trues2/total2 #computing test accuracy
    print(f'F2 Model Test Accuracy: {accuracy2}')
    print(f'F2 Model Test Loss: {total_tloss2}')
    
    params2 = sum(param.numel() for param in f2.parameters()) #computing number of parameters using .numel from pytorch, source: https://docs.pytorch.org/docs/stable/generated/torch.numel.html
    print(f"F2 Model's Number of Parameters: {params2}")

    plt.show()
    
    return None
    raise NotImplementedError("Your Code Goes Here")


if __name__ == "__main__":
    main()
