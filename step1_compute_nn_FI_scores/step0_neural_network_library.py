import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from copy import deepcopy as COPY
import pdb
import sys

def get_batchable_col_indices(X, min_proportion):
    batchable_indices = []
    for i in range(len(X[0])):
        unique_elements, counts = np.unique(X[:, i], return_counts = True)
        if (len(X) - np.max(counts))/len(X) >= min_proportion:
            batchable_indices.append(i)
    return(np.array(batchable_indices))

class ffn(torch.nn.Module):

    def __init__(self, input_len, hidden_len, num_layers, X_train):
        super(ffn, self).__init__()

        self.layers = nn.Sequential()
        self.layer_dims = [input_len] + [hidden_len]*num_layers + [1]
        for i in range(num_layers + 1):
            self.layers.add_module("Linear" + str(i + 1), 
                                   nn.Linear(self.layer_dims[i], 
                                             self.layer_dims[i + 1]))
            if i < num_layers:
                self.layers.add_module("L_ReLU" + str(i + 1), nn.LeakyReLU(0.1))
        self.mean = torch.mean(X_train, axis = 0)
        self.std = torch.std(X_train, axis = 0)
        self.sig = nn.Sigmoid()

    def forward(self, X):
        return(self.sig(self.layers((X - self.mean)/self.std)))

def get_batches(X_train, X_test, Y_train, Y_test, batch_size):

    sample_size_train = len(X_train)
    num_divisible_train = (sample_size_train - sample_size_train%batch_size)
    num_batches = int(num_divisible_train/batch_size)
    batch_indices_train = np.arange(sample_size_train)
    np.random.shuffle(batch_indices_train)
    batch_indices_train = batch_indices_train[:num_divisible_train]

    sample_size_test = len(X_test)
    batch_size_test = int(sample_size_test/num_batches)
    remainder1 = sample_size_test%batch_size_test
    remainder2 = sample_size_test%num_batches
    remainder = np.max([remainder1, remainder2])
    num_divisible_test = (sample_size_test - remainder)
    batch_indices_test = np.arange(sample_size_test)
    np.random.shuffle(batch_indices_test)
    batch_indices_test = batch_indices_test[:num_divisible_test]

    X_train_batches = X_train[batch_indices_train.reshape(num_batches, -1)]
    X_test_batches = X_test[batch_indices_test.reshape(num_batches, -1)]
    Y_train_batches = Y_train[batch_indices_train.reshape(num_batches, -1)]
    Y_test_batches = Y_test[batch_indices_test.reshape(num_batches, -1)]

    data = [X_train_batches, X_test_batches, Y_train_batches, Y_test_batches]
    # TODO: compare this part to the other function to see if combination is possible. 
    data = [[torch.tensor(data[j][i]).float() 
             for j in range(4)] for i in range(num_batches)]
    return(data, num_batches)

def train_on_batch(batch, i, hyperparameters, network, 
                   optimizer, model_name, is_binary, 
                   is_low_count_integer, col_name, alpha):

    optimizer.zero_grad()
    Bx_train, Bx_test, By_train, By_test = batch
    if is_binary:
        loss_function_train = torch.nn.BCELoss()
        loss_function_test = torch.nn.BCELoss()
    else: 
        loss_function_train = torch.nn.MSELoss()
        loss_function_test = torch.nn.MSELoss()
    training_output = network(Bx_train).reshape(-1)
    testing_output = network(Bx_test).reshape(-1)
    train_loss = loss_function_train(training_output, By_train)
    test_loss = loss_function_test(testing_output, By_test)

    # In the binary case, estimates of 0 and 1 often become so close together
    # that spearman R adds unnecessary noise due to insignificant differences
    # Pearson R compares actual values, making it better
    # and there are no outliers to worry about in the binary case. 
    # This generalizes to the low integer count case.
    if is_low_count_integer:
        train_efficacy = pearsonr(training_output.detach().numpy(), 
                                   By_train.detach().numpy())[0]
        test_efficacy = pearsonr(testing_output.detach().numpy(), By_test.detach().numpy())[0]
        if is_binary:
            y_pred = torch.round(testing_output).detach().numpy()
            y_real = By_test.detach().numpy()
            correctly_predicted_positives = np.logical_and(y_pred == 1, y_real == 1)
            correctly_predicted_negatives = np.logical_and(y_pred == 0, y_real == 0)
            sensitivity = np.sum(correctly_predicted_positives)/np.sum(y_real == 1)
            specificity = np.sum(correctly_predicted_negatives)/np.sum(y_real == 0)
            accuracy = np.sum(y_pred == y_real)/len(y_real)

    else:
        train_efficacy = spearmanr(training_output.detach().numpy(), 
                                   By_train.detach().numpy())[0]
        test_efficacy = spearmanr(testing_output.detach().numpy(), 
                                  By_test.detach().numpy())[0]
        sensitivity = np.nan
        specificity = np.nan
        accuracy = np.nan

    batch_losses = [train_loss.item(), test_loss.item(),
                    train_efficacy, test_efficacy,
                    sensitivity, specificity, accuracy]
    train_loss.backward()
    optimizer.step()

    return(batch_losses)

def train_network(data, hyperparameters, network, 
                  model_name, is_binary,
                  is_low_count_integer, col_name, alpha):

    max_epochs, batch_size, learning_rate, lr_coefs = hyperparameters
    lr_coefs = lr_coefs + [lr_coefs[-1]]
    epochs = 0
    num_non_improvements = 0
    optimizer = torch.optim.Adam(network.parameters(), 
                                 lr = learning_rate*lr_coefs[0],
                                 weight_decay = 0)  
    
    train_loss_array = []
    test_loss_array = []
    train_efficacy_array = []
    test_efficacy_array = []
    old_mean = 0
    successive_means = []
    while epochs < max_epochs and num_non_improvements < (len(lr_coefs) - 1):

        train_test_set, num_batches = get_batches(*data)
        args = [hyperparameters, network, optimizer, 
                model_name, is_binary, is_low_count_integer, col_name, alpha]
        epoch_losses = np.array([train_on_batch(batch, i, *args) 
                                 for i, batch in enumerate(train_test_set)]).T

        train_loss_array += epoch_losses[0].tolist()
        test_loss_array += epoch_losses[1].tolist()
        train_efficacy_array += epoch_losses[2].tolist()
        test_efficacy_array += epoch_losses[3].tolist()
        
        epochs += 1
        batch_efficacies = epoch_losses[3, np.isnan(epoch_losses[3]) == False]
        batch_sensitivities = epoch_losses[4, np.isnan(epoch_losses[4]) == False]
        batch_specificities = epoch_losses[5, np.isnan(epoch_losses[5]) == False]
        batch_accuracies = epoch_losses[6, np.isnan(epoch_losses[6]) == False]
        new_mean = np.mean(batch_efficacies)  
        successive_means.append(new_mean)   
  
        if new_mean < old_mean or new_mean > 0.9999:
           num_non_improvements += 1
           lr_coef = lr_coefs[num_non_improvements]
           optimizer = torch.optim.Adam(network.parameters(), 
                                        lr = learning_rate*lr_coef)

        old_mean = np.mean(batch_efficacies)
        
    output = [np.mean(batch_efficacies), np.mean(batch_sensitivities), 
              np.mean(batch_specificities), np.mean(batch_accuracies)]
    return(output)

def retrain_network(data, hyperparameters, network, 
                    model_name, is_binary, ffn,
                    is_low_count_integer, col_name,
                    ffn_params, alpha):

    starting_hps = COPY(hyperparameters)
    count = 0
    very_old_efficacy = 0
    old_efficacy = 0
    new_efficacy = 0.01
    very_old_effectiveness = [0,0,0,0]
    old_effectiveness = [0,0,0,0]
    new_effectiveness = [0,0,0,0]
    while new_efficacy - old_efficacy > 0.002:
        if count >= 10:
            print("exiting: the network's gradient descent will not stably converge.")
            exit()
        net_copy = COPY(network)
        very_old_efficacy = COPY(old_efficacy)
        very_old_effectiveness = COPY(old_effectiveness)
        old_efficacy = COPY(new_efficacy)
        old_effectiveness = COPY(new_effectiveness)
        new_effectiveness = train_network(data, hyperparameters, 
                                          net_copy, model_name, is_binary,
                                          is_low_count_integer, col_name, alpha)
        new_efficacy = new_effectiveness[0]
        if new_efficacy - old_efficacy < -0.01:
            if len(hyperparameters[3]) == 1:
                count += 1
                very_old_efficacy = 0
                old_efficacy = 0
                new_efficacy = 0.01
                very_old_effectiveness = [0,0,0,0]
                old_effectiveness = [0,0,0,0]
                new_effectiveness = [0,0,0,0]
                network = ffn(*ffn_params)
            else:
                hyperparameters[3] = hyperparameters[3][1:]
                new_efficacy = COPY(old_efficacy)
                old_efficacy = COPY(very_old_efficacy)
                new_effectiveness = COPY(old_effectiveness)
                old_effectiveness = COPY(very_old_effectiveness)
        elif new_efficacy - old_efficacy > 0.001:
            network = COPY(net_copy)

    efficacies = [old_efficacy, new_efficacy]
    effectiveness_sets = [old_effectiveness, new_effectiveness]
    networks = [network, net_copy]
    best_net_index = np.argmax([old_efficacy, new_efficacy])
    best_efficacy = efficacies[best_net_index]
    best_effectiveness = effectiveness_sets[best_net_index]
    best_network = networks[best_net_index]

    message = "model = " + model_name + ", field = " + col_name + ", r = "
    message += str(best_efficacy) + "."
    print(message)
    return(best_effectiveness, best_network)

def retrain_many_networks(data, hyperparameters, network, 
                          model_name, is_binary, ffn,
                          is_low_count_integer, col_name,
                          ffn_params, alpha, num_nets):

    starting_hyperparameters = COPY(hyperparameters)
    networks = []
    effectiveness_scores = []
    for i in range(num_nets):
        next_effectiveness, next_network = retrain_network(data, hyperparameters, network, 
                                                           model_name, is_binary, ffn,
                                                           is_low_count_integer, col_name,
                                                           ffn_params, alpha)
        networks.append(next_network)
        effectiveness_scores.append(next_effectiveness)
        network = ffn(*ffn_params)
        hyperparameters = COPY(starting_hyperparameters)
    best_net_index = np.argmax(effectiveness_scores, axis = 0)[0]
    return(effectiveness_scores[best_net_index], networks[best_net_index])    

def validate_fold(data, indices, main_col_name, num_layers, alpha): 

    X = data[0]
    y = data[1]
    hyperparameters_simple = data[2]
    hyperparameters_network = data[3]
    train_index, test_index = indices
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_tensor = torch.tensor(X_train).float()
    base_params = [len(X[0]), 50, num_layers, X_tensor]
    network_params = [len(X[0]), 50, num_layers + 1, X_tensor]
    base_model = ffn(*base_params)
    network = ffn(*network_params)
    data = [X_train, X_test, y_train, y_test, 1000]
    if len(np.unique(y)) == 2: is_binary = True
    else: is_binary = False
    if len(np.unique(y)) <= 10: is_low_count_integer = True
    else: is_low_count_integer = False
    
    simple_output = retrain_network(data, hyperparameters_simple, 
                                    base_model, "simple model", is_binary, 
                                    ffn, is_low_count_integer,
                                    main_col_name, base_params, alpha)
    simple_model_effectiveness = simple_output[0]
    base_model = simple_output[1]
 
    network_output = retrain_network(data, hyperparameters_network, 
                                     network, "network", is_binary, 
                                     ffn, is_low_count_integer, 
                                     main_col_name, network_params, alpha)
    neural_network_effectiveness = network_output[0]
    network = network_output[1]
    return(np.array([simple_model_effectiveness, neural_network_effectiveness]))

def k_fold_CV(data, k, main_col_name, num_layers, alpha):
    X = data[0]
    y = data[1]
    kf = KFold(n_splits = k, shuffle = True)
    kf.get_n_splits(X)
    efficacies =  np.array([validate_fold(data, indices, main_col_name, 
                                          num_layers, alpha)
                            for indices in kf.split(X)]) 
    simple_model_effectiveness = efficacies[:, 0, :]
    neural_network_effectiveness = efficacies[:, 1, :]
    print(np.mean(simple_model_effectiveness, axis = 0))
    return([simple_model_effectiveness, neural_network_effectiveness])

