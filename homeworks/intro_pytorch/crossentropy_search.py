if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from losses import CrossEntropyLossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from .optimizers import SGDOptimizer
    from .losses import CrossEntropyLossLayer
    from .train import plot_model_guesses, train

from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def crossentropy_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the CrossEntropy problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU activation function after each hidden layer
    NOTE: Each model should end with a Softmax layer due to CrossEntropyLossLayer requirement.

    Notes:
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.

    Args:
        dataset_train (TensorDataset): Dataset for training.
        dataset_val (TensorDataset): Dataset for validation.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    training_dict = {'Linear Regression': None, '1 HL NN with Sigmoid': None, '1 HL NN with ReLU': None,
                     '2 HL NN with Sigmoid & ReLU': None, '2 HL NN with ReLU & Sigmoid': None,
                     '2 HL NN with ReLU': None} #loading a dictionary for each of the six models where the values will be replaced by each of their training and val loss history dicts
    train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True) #applying DataLoader function provided at the top to training and val in these two lines
    val_dataloader = DataLoader(dataset_val, batch_size=64, shuffle=True)
    crit = CrossEntropyLossLayer() #our loss criterion is CrossEntropy Loss
    
    tr_dim = dataset_train.tensors[0].shape[1]
    dim_out = 2  #two classes will be the potential output for the models

    #Model 1: Linear Regression
    model1 = nn.Sequential( #since our output needs to be softmax, we build a neural network using nn.Sequential (note about nn.Sequential is at the end of the function)
        LinearLayer(tr_dim, dim_out, RNG), #creating the linear with RNG provided above as the generator parameter and it will output 2 units
        SoftmaxLayer()
    )
    opt1 = SGDOptimizer(model1.parameters(), lr = 0.01) #setting the optimizer for this LR model with learning rate of 0.01 arbitrarily
    train_dict1 = train(train_loader=train_dataloader, model = model1, criterion = crit, optimizer = opt1, 
                        val_loader = val_dataloader) #obtaining the training and validation loss history for model 1 (linear regression)
    train_dict1['model'] = model1 #adding a key:value to the train dict of the model:its architecture
    training_dict["Linear Regression"] = train_dict1 #appending this training dict with the three keys (train,val,model) to the training dict as the value for key 1

    #Model 2: Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
    model2 = nn.Sequential( #we use nn.Sequential here to compose this neural network with one hidden layer and a sigmoid activation function
        LinearLayer(tr_dim, 2, RNG), #we make the hidden layer a linear regression (not for a strict reason, can technically be any function)
        SigmoidLayer(), #the layer following the first hidden layer is the sigmoid as specified
        LinearLayer(2, 2, RNG), #applying linear regression to obtain outputs before we perform softmax
        SoftmaxLayer() #making the output a softmax
    )
    opt2 = SGDOptimizer(model2.parameters(), lr = 0.01) #repeating same process for model 1
    train_dict2 = train(train_loader=train_dataloader, model = model2, criterion = crit, optimizer = opt2, 
                        val_loader = val_dataloader)
    train_dict2['model'] = model2
    training_dict["1 HL NN with Sigmoid"] = train_dict2

    #Model 3: Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
    model3 = nn.Sequential( #we use nn.Sequential here to compose this neural network with one hidden layer and a ReLU activation function
        LinearLayer(tr_dim, 2, RNG), #we make the hidden layer a linear regression (not for a strict reason, can technically be any function)
        ReLULayer(), #the layer following the first hidden layer is the ReLU as specified
        LinearLayer(2, 2, RNG), #applying linear regression to obtain outputs before we perform softmax
        SoftmaxLayer() #making the output a softmax
    )
    opt3 = SGDOptimizer(model3.parameters(), lr = 0.01) #repeating same process for model 1
    train_dict3 = train(train_loader=train_dataloader, model = model3, criterion = crit, optimizer = opt3, 
                        val_loader = val_dataloader)
    train_dict3['model'] = model3
    training_dict["1 HL NN with ReLU"] = train_dict3

    #Model 4: Network with two hidden layers (each with size 2) and Sigmoid, ReLU activation functions after corresponding hidden layers
    model4 = nn.Sequential( #we use nn.Sequential here to compose this neural network with two hidden layers, a Sigmoid activation function, and a ReLU activation function
        LinearLayer(tr_dim, 2, RNG), #we make the hidden layer a linear regression (not for a strict reason, can technically be any function)
        SigmoidLayer(), #the layer following the first hidden layer is the sigmoid as specified
        LinearLayer(2, 2, RNG), #using linear regression for second hidden layer as well
        ReLULayer(), #the layer following the second hidden layer is the sigmoid as specified
        LinearLayer(2, 2, RNG), #applying linear regression to obtain outputs before we perform softmax
        SoftmaxLayer() #making the output a softmax
    )
    opt4 = SGDOptimizer(model4.parameters(), lr = 0.01) #repeating same process for model 1
    train_dict4 = train(train_loader=train_dataloader, model = model4, criterion = crit, optimizer = opt4, 
                        val_loader = val_dataloader)
    train_dict4['model'] = model4
    training_dict['2 HL NN with Sigmoid & ReLU'] = train_dict4

    #Model 5: Network with two hidden layers (each with size 2) and ReLU, Sigmoid activation functions after corresponding hidden layers
    model5 = nn.Sequential( #we use nn.Sequential here to compose this neural network with two hidden layers, a ReLU activation function, and a Sigmoid activation function
        LinearLayer(tr_dim, 2, RNG), #we make the hidden layer a linear regression (not for a strict reason, can technically be any function)
        ReLULayer(), #the layer following the first hidden layer is the ReLU as specified
        LinearLayer(2, 2, RNG), #using linear regression for second hidden layer as well
        SigmoidLayer(), #the layer following the second hidden layer is the sigmoid as specified
        LinearLayer(2, 2, RNG), #applying linear regression to obtain outputs before we perform softmax
        SoftmaxLayer() #making the output a softmax
    )
    opt5 = SGDOptimizer(model5.parameters(), lr = 0.01) #repeating same process for model 1
    train_dict5 = train(train_loader=train_dataloader, model = model5, criterion = crit, optimizer = opt5, 
                        val_loader = val_dataloader)
    train_dict5['model'] = model5
    training_dict['2 HL NN with ReLU & Sigmoid'] = train_dict5

    #Model 6: Network with two hidden layers (each with size 2) and ReLU activation function after each hidden layer
    model6 = nn.Sequential( #we use nn.Sequential here to compose this neural network with two hidden layers and a ReLU activation function after each hidden layer
        LinearLayer(tr_dim, 2, RNG), #we make the hidden layer a linear regression (not for a strict reason, can technically be any function)
        ReLULayer(), #the layer following the first hidden layer is the ReLU as specified
        LinearLayer(2, 2, RNG), #using linear regression for second hidden layer as well
        ReLULayer(), #the layer following the second hidden layer is the ReLU as specified
        LinearLayer(2, 2, RNG), #applying linear regression to obtain outputs before we perform softmax
        SoftmaxLayer() #making the output a softmax
    )
    opt6 = SGDOptimizer(model6.parameters(), lr = 0.01) #repeating same process for model 1
    train_dict6 = train(train_loader=train_dataloader, model = model6, criterion = crit, optimizer = opt6, 
                        val_loader = val_dataloader)
    train_dict6['model'] = model6
    training_dict['2 HL NN with ReLU'] = train_dict6

    #nn.Sequential is taken from Section 6 notes/notebook: https://colab.research.google.com/drive/1gi3urpWI8sprG76Bj_ZMux3U8vh7N31R?usp=share_link%22#scrollTo=5U9Xi9AmJhxP
    #I could not figure a different to define neural networks without creating all new classes for each of the six models, hence why I use nn.Sequential (which from what I can tell is not prohibited)

    return training_dict #returning the training dict with training and val loss history for all six models
    raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def accuracy_score(model, dataloader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for CrossEntropy.
            Each example is a tuple consisting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is an integer representing a correct class to a corresponding observation.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to MSE accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    with torch.no_grad(): #using torch.no_grad for computing accuracy (refer to train.py for details)
        trues = 0 
        total = 0
        for obs,targets in dataloader:
            outputs = model(obs)
            preds = outputs.argmax(dim=1) #obtaining the prediction with the highest probability, source: https://docs.pytorch.org/docs/stable/generated/torch.argmax.html
            trues += (preds == targets).sum().item() #summing all predictions that are correct per batch
            total += targets.size(0) #incrementing the total number of targets 
        
    accuracy = trues/total #calculating accuracy = (TP + TN)/Total
    return accuracy
    raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A", start_line=7)
def main():
    """
    Main function of the Crossentropy problem.
    It should:
        1. Call crossentropy_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 12 lines total).
            x-axis should be epochs, y-axis should me Crossentropy loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y))
    dataset_val = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    dataset_test = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))

    ce_configs = crossentropy_parameter_search(dataset_train, dataset_val)
    plt.figure(1, figsize = (12,8)) #defining our first plot for CE Loss
    for model_name, model in ce_configs.items(): #iterating through dict for CE configurations to obtain each of the six model's names (keys) and training+val loss history (values)
        plt.plot(model['train'], label = f'{model_name} Training Loss')
        plt.plot(model['val'], label = f'{model_name} Validation Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('CrossEntropy Loss')
    plt.legend()

    #now we'll find which model has the smallest validation loss, determining the best one
    #creating empty lists for models, their architectures, and each one's minimum validation loss
    model_names = []
    models = []
    min_vlosses = []

    for model_name, _model in ce_configs.items(): #iterating through CE configs dict to append each of these three aspects for the models to the lists above
        model_vloss = torch.tensor(_model['val']) #creating a tensor out of the values for the model's validation loss
        model_minvl = torch.min(model_vloss) #computing the minimum val loss using torch.min, source: https://docs.pytorch.org/docs/stable/generated/torch.min.html
        model_names.append(model_name)
        models.append(_model['model'])
        min_vlosses.append(model_minvl)
    
    min_vlosses = torch.tensor(min_vlosses) #converting the list of minimum val losses for each model to a tensor
    
    min_vloss = torch.min(min_vlosses) #find the minimum validation loss out of all six models', source: https://docs.pytorch.org/docs/stable/generated/torch.min.html
    i_star = torch.argmin(min_vlosses) #finding the index of the minimum val loss to obtain the corresponding model and name below, source: https://docs.pytorch.org/docs/stable/generated/torch.argmin.html
    model_star = models[i_star]
    name_star = model_names[i_star]

    print(f'The best model in terms of validation CE Loss is {name_star}')
    print(f'The best model architecture is {model_star}, and its validation loss is {min_vloss:.4f}')

    test_dataloader = DataLoader(dataset_test) #converting test set into a dataloader
    accuracy = accuracy_score(model_star, test_dataloader) #applying accuracy score function to obtain test accuracy
    print(f'Best Model Testing Accuracy: {accuracy}')

    plt.figure(2,figsize = (10,5)) #creating second plot figure for the best model's guesses
    plot_model_guesses(test_dataloader,model_star, title = f'Best CE Loss Model Guesses for {name_star}')

    return None
    raise NotImplementedError("Your Code Goes Here")


if __name__ == "__main__":
    main()
