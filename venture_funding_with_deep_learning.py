#!/usr/bin/env python
# coding: utf-8

# # Venture Funding with Deep Learning
# 
# You work as a risk management associate at Alphabet Soup, a venture capital firm. Alphabet Soup’s business team receives many funding applications from startups every day. This team has asked you to help them create a model that predicts whether applicants will be successful if funded by Alphabet Soup.
# 
# The business team has given you a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. With your knowledge of machine learning and neural networks, you decide to use the features in the provided dataset to create a binary classifier model that will predict whether an applicant will become a successful business. The CSV file contains a variety of information about these businesses, including whether or not they ultimately became successful.
# 
# ## Instructions:
# 
# The steps for this challenge are broken out into the following sections:
# 
# * Prepare the data for use on a neural network model.
# 
# * Compile and evaluate a binary classification model using a neural network.
# 
# * Optimize the neural network model.
# 
# ### Prepare the Data for Use on a Neural Network Model 
# 
# Using your knowledge of Pandas and scikit-learn’s `StandardScaler()`, preprocess the dataset so that you can use it to compile and evaluate the neural network model later.
# 
# Open the starter code file, and complete the following data preparation steps:
# 
# 1. Read the `applicants_data.csv` file into a Pandas DataFrame. Review the DataFrame, looking for categorical variables that will need to be encoded, as well as columns that could eventually define your features and target variables.   
# 
# 2. Drop the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame, because they are not relevant to the binary classification model.
#  
# 3. Encode the dataset’s categorical variables using `OneHotEncoder`, and then place the encoded variables into a new DataFrame.
# 
# 4. Add the original DataFrame’s numerical variables to the DataFrame containing the encoded variables.
# 
# > **Note** To complete this step, you will employ the Pandas `concat()` function that was introduced earlier in this course. 
# 
# 5. Using the preprocessed data, create the features (`X`) and target (`y`) datasets. The target dataset should be defined by the preprocessed DataFrame column “IS_SUCCESSFUL”. The remaining columns should define the features dataset. 
# 
# 6. Split the features and target sets into training and testing datasets.
# 
# 7. Use scikit-learn's `StandardScaler` to scale the features data.
# 
# ### Compile and Evaluate a Binary Classification Model Using a Neural Network
# 
# Use your knowledge of TensorFlow to design a binary classification deep neural network model. This model should use the dataset’s features to predict whether an Alphabet Soup&ndash;funded startup will be successful based on the features in the dataset. Consider the number of inputs before determining the number of layers that your model will contain or the number of neurons on each layer. Then, compile and fit your model. Finally, evaluate your binary classification model to calculate the model’s loss and accuracy. 
#  
# To do so, complete the following steps:
# 
# 1. Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.
# 
# > **Hint** You can start with a two-layer deep neural network model that uses the `relu` activation function for both layers.
# 
# 2. Compile and fit the model using the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric.
# 
# > **Hint** When fitting the model, start with a small number of epochs, such as 20, 50, or 100.
# 
# 3. Evaluate the model using the test data to determine the model’s loss and accuracy.
# 
# 4. Save and export your model to an HDF5 file, and name the file `AlphabetSoup.h5`. 
# 
# ### Optimize the Neural Network Model
# 
# Using your knowledge of TensorFlow and Keras, optimize your model to improve the model's accuracy. Even if you do not successfully achieve a better accuracy, you'll need to demonstrate at least two attempts to optimize the model. You can include these attempts in your existing notebook. Or, you can make copies of the starter notebook in the same folder, rename them, and code each model optimization in a new notebook. 
# 
# > **Note** You will not lose points if your model does not achieve a high accuracy, as long as you make at least two attempts to optimize the model.
# 
# To do so, complete the following steps:
# 
# 1. Define at least three new deep neural network models (the original plus 2 optimization attempts). With each, try to improve on your first model’s predictive accuracy.
# 
# > **Rewind** Recall that perfect accuracy has a value of 1, so accuracy improves as its value moves closer to 1. To optimize your model for a predictive accuracy as close to 1 as possible, you can use any or all of the following techniques:
# >
# > * Adjust the input data by dropping different features columns to ensure that no variables or outliers confuse the model.
# >
# > * Add more neurons (nodes) to a hidden layer.
# >
# > * Add more hidden layers.
# >
# > * Use different activation functions for the hidden layers.
# >
# > * Add to or reduce the number of epochs in the training regimen.
# 
# 2. After finishing your models, display the accuracy scores achieved by each model, and compare the results.
# 
# 3. Save each of your models as an HDF5 file.
# 

# In[ ]:


# Imports
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder

import warnings
warnings.filterwarnings('ignore')


# ---
# 
# ## Prepare the data to be used on a neural network model

# ### Step 1: Read the `applicants_data.csv` file into a Pandas DataFrame. Review the DataFrame, looking for categorical variables that will need to be encoded, as well as columns that could eventually define your features and target variables.  
# 

# In[ ]:


# Read the applicants_data.csv file from the Resources folder into a Pandas DataFrame
applicant_data_df =  pd.read_csv(Path('./Resources/applicants_data.csv'))

# Review the DataFrame
print("applicants.csv data frame head:")
display(applicant_data_df.head())


# In[ ]:


# Review the data types associated with the columns
display(applicant_data_df.dtypes)


# In[ ]:


# Degug checkpoint
display(applicant_data_df.shape)


# ### Step 2: Drop the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame, because they are not relevant to the binary classification model.

# In[ ]:


# Drop the 'EIN' and 'NAME' columns from the DataFrame
applicant_data_df = applicant_data_df.drop(columns=['EIN', 'NAME'])

# Review the DataFrame
print("applicants.csv data frame head:")
display(applicant_data_df.head())


# ### Step 3: Encode the dataset’s categorical variables using `OneHotEncoder`, and then place the encoded variables into a new DataFrame.

# In[ ]:


# Degug checkpoint
display(applicant_data_df.shape)


# In[ ]:


# Create a list of categorical variables 
applicant_categorical_col_lst = list(applicant_data_df.dtypes[applicant_data_df.dtypes == "object"].index)

# Display the categorical variables list
print("applicant_categorical_col_lst:")
display(applicant_categorical_col_lst)


# In[ ]:


# Create a OneHotEncoder instance
one_hot_enc_inst = OneHotEncoder(sparse=False)


# In[ ]:


# Degug checkpoint
display(type(one_hot_enc_inst))


# In[ ]:


# Encode the categorcal variables using OneHotEncoder
applicant_data_categorical_onehotenc_npa = one_hot_enc_inst.fit_transform(applicant_data_df[applicant_categorical_col_lst])


# In[ ]:


# Degug checkpoint
display(type(applicant_data_categorical_onehotenc_npa))
display(len(applicant_data_categorical_onehotenc_npa))  # rows
display(len(applicant_data_categorical_onehotenc_npa[0]))  # columns
display(applicant_data_categorical_onehotenc_npa)


# In[ ]:


display(applicant_categorical_col_lst)


# In[ ]:


applicant_categorical_onehotenc_col_lst = one_hot_enc_inst.get_feature_names(applicant_categorical_col_lst)


# In[ ]:


# Degug checkpoint
display(applicant_categorical_onehotenc_col_lst.shape)
display(applicant_categorical_onehotenc_col_lst)


# In[ ]:


# Create a DataFrame with the encoded variables
applicant_data_categorical_onehotenc_df = pd.DataFrame(
    applicant_data_categorical_onehotenc_npa,
    columns = applicant_categorical_onehotenc_col_lst
)

# Review the DataFrame
print("applicant_data_categorical_onehotenc_df head:")
display(applicant_data_categorical_onehotenc_df.head())


# In[ ]:


# Degug checkpoint
display(applicant_data_categorical_onehotenc_df.shape)


# In[ ]:


# Create a DataFrame with the columnns containing numerical variables from the original dataset
applicant_data_numerical_df = applicant_data_df.drop(columns = applicant_categorical_col_lst)


# In[ ]:


# Degug checkpoint
print("applicant_data_numerical_df head:")
display(applicant_data_numerical_df.shape)
display(applicant_data_numerical_df.head())


# ### Step 4: Add the original DataFrame’s numerical variables to the DataFrame containing the encoded variables.
# 
# > **Note** To complete this step, you will employ the Pandas `concat()` function that was introduced earlier in this course. 

# In[ ]:


# Add the numerical variables from the original DataFrame to the one-hot encoding DataFrame
applicant_data_numerical_onehotenc_df = pd.concat(
    [
        applicant_data_numerical_df,
        applicant_data_categorical_onehotenc_df
    ],
    axis=1
)

# Review the Dataframe
print("applicant_data_numerical_onehotenc_df:")
display(applicant_data_numerical_onehotenc_df.head())


# In[ ]:


# Degug checkpoint
display(applicant_data_numerical_onehotenc_df.shape)


# ### Step 5: Using the preprocessed data, create the features (`X`) and target (`y`) datasets. The target dataset should be defined by the preprocessed DataFrame column “IS_SUCCESSFUL”. The remaining columns should define the features dataset. 
# 
# 

# In[ ]:


# Define features set X by selecting all columns but IS_SUCCESSFUL
X_df = applicant_data_numerical_onehotenc_df.drop(columns=['IS_SUCCESSFUL'])

# Review the features DataFrame
print("X_df:")
display(X_df.head())


# In[ ]:


# Degug checkpoint
display(X_df.shape)


# In[ ]:


# Define the target set y using the IS_SUCCESSFUL column
y_srs = applicant_data_numerical_onehotenc_df["IS_SUCCESSFUL"]

# Display a sample of y
print("target set y:")
display(type(y_srs))
display(y_srs)


# In[ ]:


# Degug checkpoint
display(y_srs.shape)


# ### Step 6: Split the features and target sets into training and testing datasets.
# 

# In[ ]:


# Split the preprocessed data into a training and testing dataset
# Assign the function a random_state equal to 1
X_train_df, X_test_df, y_train_srs, y_test_srs = train_test_split(X_df, y_srs, random_state=1)


# In[ ]:


# Data checkpoint
display(type(X_train_df))
display(type(X_test_df))
display(type(y_train_srs))
display(type(y_test_srs))

display(X_train_df.shape)
display(X_test_df.shape)
display(y_train_srs.shape)
display(y_test_srs.shape)


# ### Step 7: Use scikit-learn's `StandardScaler` to scale the features data.

# In[ ]:


# Create a StandardScaler instance
standard_scaler_inst = StandardScaler()

# Fit the scaler to the features training dataset
X_train_scaler = standard_scaler_inst.fit(X_train_df)

# Fit the scaler to the features training dataset
X_train_scaled_npa = X_train_scaler.transform(X_train_df)
X_test_scaled_npa = X_train_scaler.transform(X_test_df)


# In[ ]:


# Data checkpoint
display(X_train_scaled_npa.shape)
display(X_test_scaled_npa.shape)


# In[ ]:


# Convert back to dataframe (to make it easier to drop feature sets as part of evaluating different model permuations below)
X_train_scaled_df = pd.DataFrame(X_train_scaled_npa, columns = X_df.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled_npa, columns = X_df.columns)


# In[ ]:


# Data checkpoint
display(X_train_scaled_df.shape)
display(X_test_scaled_df.shape)


# ---
# 
# ## Compile and Evaluate a Binary Classification Model Using a Neural Network

# ### Step 1: Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.
# 
# > **Hint** You can start with a two-layer deep neural network model that uses the `relu` activation function for both layers.
# 

# In[ ]:


# Define the the number of inputs (features) to the model
number_input_features = X_train_scaled_df.shape[1]

# Review the number of features
print("Number of input features = ", number_input_features)


# In[ ]:


# Define the number of neurons in the output layer
number_output_neurons = 1


# In[ ]:


# Define the number of hidden nodes for the first hidden layer
num_hidden_nodes_layer1 =  (number_input_features + 1) // 2 

# Review the number hidden nodes in the first layer
print("Number of hidden nodes in layer 1 = ", num_hidden_nodes_layer1)


# In[ ]:


# Define the number of hidden nodes for the second hidden layer
num_hidden_nodes_layer2 =  (num_hidden_nodes_layer1 + 1) // 2

# Review the number hidden nodes in the second layer
print("Number of hidden nodes in layer 2 = ", num_hidden_nodes_layer2)


# In[ ]:


# Create the Sequential model instance
nn_sequential_original_model = Sequential()


# In[ ]:


# Add the first hidden layer
nn_sequential_original_model.add(Dense(units=num_hidden_nodes_layer1, activation="relu", input_dim=number_input_features))


# In[ ]:


# Add the second hidden layer
nn_sequential_original_model.add(Dense(units=num_hidden_nodes_layer2, activation="relu"))


# In[ ]:


# Add the output layer to the model specifying the number of output neurons and activation function
nn_sequential_original_model.add(Dense(units=number_output_neurons, activation="sigmoid"))


# In[ ]:


# Display the Sequential model summary
nn_sequential_original_model.summary()


# ### Step 2: Compile and fit the model using the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric.
# 

# In[ ]:


# Compile the Sequential model
nn_sequential_original_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[ ]:


# Fit the model using 50 epochs and the training data
nn_sequential_original_model_fit = nn_sequential_original_model.fit(X_train_scaled_df, y_train_srs, epochs=50)


# ### Step 3: Evaluate the model using the test data to determine the model’s loss and accuracy.
# 

# In[ ]:


# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
nn_sequential_permutation_model_loss, nn_sequential_permutation_model_accuracy =  nn_sequential_original_model.evaluate(X_test_scaled_df,y_test_srs,verbose=2)

# Display the model loss and accuracy results
print(f"Loss: {nn_sequential_permutation_model_loss}, Accuracy: {nn_sequential_permutation_model_accuracy}")


# ### Step 4: Save and export your model to an HDF5 file, and name the file `AlphabetSoup.h5`. 
# 

# In[ ]:


# Set the model's file path
file_path = Path("./Resources/AlphabetSoup.h5")

# Export your model to a HDF5 file
nn_sequential_original_model.save_weights(file_path)    


# ---
# 
# ## Optimize the neural network model
# 

# ### Step 1: Define at least three new deep neural network models (resulting in the original plus 3 optimization attempts). With each, try to improve on your first model’s predictive accuracy.
# 
# > **Rewind** Recall that perfect accuracy has a value of 1, so accuracy improves as its value moves closer to 1. To optimize your model for a predictive accuracy as close to 1 as possible, you can use any or all of the following techniques:
# >
# > * Adjust the input data by dropping different features columns to ensure that no variables or outliers confuse the model.
# >
# > * Add more neurons (nodes) to a hidden layer.
# >
# > * Add more hidden layers.
# >
# > * Use different activation functions for the hidden layers.
# >
# > * Add to or reduce the number of epochs in the training regimen.
# 

# ### Alternate Models

# In[ ]:


# Create a list of activation functions that will permutate over
activation_function_lst = ["relu", "sigmoid", "exponential", "swish"]
num_activation_functions = len(activation_function_lst)


# In[ ]:


# Debug checkpoint
display(num_activation_functions)
display(activation_function_lst)


# In[ ]:


# Create a copy of the categorical feature list  (will iterate through and drop one column at a time to determine if removing the feature set impacts the model result)
model_feature_perm_lst = applicant_categorical_col_lst.copy()
model_feature_perm_lst.insert(0,'NO_FEATURE_DROPS')  # Add a dummy "No Filter" to not match any column names - and thus preserving the full original list (i.e. no columns drops)
num_model_feature_perm_lst = len(model_feature_perm_lst)


# In[ ]:


# Debug checkpoint
display(num_model_feature_perm_lst)
display(model_feature_perm_lst)


# In[ ]:


# Setup neural network model permutation parameters

# Permuatate through the feature set (drop one feature column per permuation)
strt_num_model_feature_perm_lst_idx = 0
end_num_model_feature_perm_lst_idx = num_model_feature_perm_lst-1
inc_num_model_feature_perm_lst_idx = 1

# Permuatate over different activation lists
strt_num_activation_functions_idx = 0
end_num_activation_functions_idx = num_activation_functions-1
inc_num_activation_functions_idx = 1

# Permuatate the # of hidden layers
strt_num_hidden_layers = 3
end_hidden_layers = 3
inc_hidden_layers = 3

# Permuatate the # of nodes in hidden layer 1
strt_num_hidden_nodes_layer1 = 64
end_num_hidden_nodes_layer1 = 64
inc_num_hidden_nodes_layer1 = 64

# Permuatate the # of nodes in hidden layer 2
strt_num_hidden_nodes_layer2 = 24
end_num_hidden_nodes_layer2 = 24
inc_num_hidden_nodes_layer2 = 24

# Permuatate the # of nodes in hidden layer 3
strt_num_hidden_nodes_layer3 = 4
end_num_hidden_nodes_layer3 = 4
inc_num_hidden_nodes_layer3 = 4

# Permuatate the # of epochs
strt_num_epochs = 5
end_num_epochs = 5
inc_num_epochs = 5


# In[ ]:


# Initialize the starting model permutation values
num_model_feature_perm_lst_idx = strt_num_model_feature_perm_lst_idx

num_hidden_layers = strt_num_hidden_layers
num_activation_functions_idx = strt_num_activation_functions_idx
num_hidden_nodes_layer1 = strt_num_hidden_nodes_layer1
num_hidden_nodes_layer2 = strt_num_hidden_nodes_layer2
num_hidden_nodes_layer3 = strt_num_hidden_nodes_layer3
num_epochs = strt_num_epochs

# Create empty lists which will store the model parameters for each permutation run
num_model_feature_perm_lst_idx_lst = []
num_activation_functions_idx_lst = []
num_hidden_layers_lst = []
num_hidden_nodes_layer1_lst = []
num_hidden_nodes_layer2_lst = []
num_hidden_nodes_layer3_lst = []
num_epochs_lst        = []

total_num_model_permutations = 0;
is_done_setting_up_model_permutations = False

print_debug = True

# Loop through each of the permutations
while (not is_done_setting_up_model_permutations):

    # Make sure the # of nodes in subsequent hidden layers is always smaller than the current hidden layer
    if ((num_hidden_nodes_layer2 < num_hidden_nodes_layer1) and (num_hidden_nodes_layer3 < num_hidden_nodes_layer2)):
        
        # Add to the list the current permuation model parameters
        num_model_feature_perm_lst_idx_lst.append(num_model_feature_perm_lst_idx)
        num_activation_functions_idx_lst.append(num_activation_functions_idx)
        num_hidden_layers_lst.append(num_hidden_layers)
        num_hidden_nodes_layer1_lst.append(num_hidden_nodes_layer1)
        num_hidden_nodes_layer2_lst.append(num_hidden_nodes_layer2)
        num_hidden_nodes_layer3_lst.append(num_hidden_nodes_layer3)
        num_epochs_lst.append(num_epochs)

        # Optional debug printing of values
        if (print_debug):
            print(f"Setting Up Model Permuation #{total_num_model_permutations+1}")
            if (num_model_feature_perm_lst_idx > 0):
                print(f"\tFeature Drop = {model_feature_perm_lst[num_model_feature_perm_lst_idx]}")
            print(f"\t Use Model Activation Function = {activation_function_lst[num_activation_functions_idx]}")                
            print(f"\t# Hidden Layers = {num_hidden_layers}")
            print(f"\t\t# Hidden Layer 1 Nodes = {num_hidden_nodes_layer1}")
            if (num_hidden_layers > 1):
                print(f"\t\t# Hidden Layer 2 Nodes = {num_hidden_nodes_layer2}")
            if (num_hidden_layers > 2):
                print(f"\t\t# Hidden Layer 3 Nodes = {num_hidden_nodes_layer3}")
            print(f"\t# Epochs = {num_epochs}")

        # Increment the total number of runs/permuataions variable
        total_num_model_permutations = total_num_model_permutations + 1  
    
    # Scan through the different permuations values,
    #    if did not reach its end/final value, increment by the defined "increment" value
    #    else reset the value to its starting value, and move on to the next permuation parameter
    # Permutation order:
    #    # of epochs
    #    # of nodes in hidden layer 3 (if # of hidden layers has 3 layers)
    #    # of nodes in hidden layer 2 (if # of hidden layers has 2 layers)
    #    # of nodes in hidden layer 1 (there will always be a hidden layer 1))
    #    # of hidden layers
    
    if (num_epochs < end_num_epochs): 
        num_epochs = num_epochs + inc_num_epochs 
    else:
        num_epochs = strt_num_epochs
        
        if ((num_hidden_layers > 2) and (num_hidden_nodes_layer3 < end_num_hidden_nodes_layer3)): 
            num_hidden_nodes_layer3 = num_hidden_nodes_layer3 + inc_num_hidden_nodes_layer3
        else:
            num_hidden_nodes_layer3 = strt_num_hidden_nodes_layer3

            if ((num_hidden_layers > 1) and (num_hidden_nodes_layer2 < end_num_hidden_nodes_layer2)): 
                num_hidden_nodes_layer2 = num_hidden_nodes_layer2 + inc_num_hidden_nodes_layer2
            else:
                num_hidden_nodes_layer2 = strt_num_hidden_nodes_layer2

                if (num_hidden_nodes_layer1 < end_num_hidden_nodes_layer1): 
                    num_hidden_nodes_layer1 = num_hidden_nodes_layer1 + inc_num_hidden_nodes_layer1
                else:
                    num_hidden_nodes_layer1 = strt_num_hidden_nodes_layer1

                    if (num_hidden_layers < end_hidden_layers): 
                        num_hidden_layers = num_hidden_layers + inc_hidden_layers
                    else:
                        num_hidden_layers = strt_num_hidden_layers

                        if (num_activation_functions_idx < end_num_activation_functions_idx): 
                            num_activation_functions_idx = num_activation_functions_idx + inc_num_activation_functions_idx
                        else:
                            num_activation_functions_idx = strt_num_activation_functions_idx
                        
                            if (num_model_feature_perm_lst_idx < end_num_model_feature_perm_lst_idx):
                                num_model_feature_perm_lst_idx = num_model_feature_perm_lst_idx + inc_num_model_feature_perm_lst_idx
                            else:                            
                                is_done_setting_up_model_permutations = True 


# In[ ]:


# Run the neural network through all the permutations

print_debug = True
model_verbose = False

# create an empty array that will hold the neural network model for each permuation
nn_sequential_permutation_model_npa = np.empty(total_num_model_permutations,  dtype=np.object)

# Keep track of model permutation with best results
best_permutation_num = 0
best_loss_value = 0
best_accuracy_value = 0

# Create an initial empty list containter to hold the fit model history and permuation/losses
nn_sequential_permutation_model_fit_lst = []
nn_sequential_permutation_model_loss_lst = []
nn_sequential_permutation_model_accuracy_lst = []

for x in range(total_num_model_permutations):

    # Check to see if should drop a feature in this model permutation, if yes, drop it
    if (num_model_feature_perm_lst_idx_lst[x] > 0):
        feature_to_drop = model_feature_perm_lst[num_model_feature_perm_lst_idx_lst[x]]
        display(feature_to_drop)
        feature_to_drop_idx = X_train_scaled_df.loc[:,X_train_scaled_df.columns.str.startswith(feature_to_drop)].columns        
        feature_to_drop_npa = np.array(feature_to_drop_idx)
        
        X_train_scaled_feature_drop_df = X_train_scaled_df.drop(columns=feature_to_drop_npa)
        X_test_scaled_feature_drop_df = X_test_scaled_df.drop(columns=feature_to_drop_npa)
    else:
        X_train_scaled_feature_drop_df = X_train_scaled_df
        X_test_scaled_feature_drop_df = X_test_scaled_df

    # Define the the number of inputs (features) to the model
    number_input_features = X_train_scaled_feature_drop_df.shape[1]
    number_output_neurons = 1

    if (print_debug):
        print(f"Running Permuation #{x+1}") 
        if (num_model_feature_perm_lst_idx_lst[x] > 0):
            print(f"\tFeature Drop = {model_feature_perm_lst[num_model_feature_perm_lst_idx_lst[x]]}")
        print(f"\t# Input Features = {number_input_features}")
        print(f"\t Use Model Activation Function = {activation_function_lst[num_activation_functions_idx_lst[x]]}")                
        print(f"\t# Hidden Layers = {num_hidden_layers_lst[x]}")
        print(f"\t# Layer 1 Nodes = {num_hidden_nodes_layer1_lst[x]}")
        if (num_hidden_layers_lst[x] > 1):
            print(f"\t# Layer 2 Nodes = {num_hidden_nodes_layer2_lst[x]}")
        if (num_hidden_layers_lst[x] > 2):
            print(f"\t# Layer 3 Nodes = {num_hidden_nodes_layer3_lst[x]}")
        print(f"\t# of Output Nuerons = {number_output_neurons}")
        print(f"\t# Epochs = {num_epochs_lst[x]}")
    
    # Create the Sequential model instance
    nn_sequential_permutation_model_npa[x] = Sequential()

    # Add the first hidden layer
    nn_sequential_permutation_model_npa[x].add(Dense(units=num_hidden_nodes_layer1_lst[x], activation=activation_function_lst[num_activation_functions_idx_lst[x]], input_dim=number_input_features))

    # If exist in this model permutation, add the second hidden layer
    if (num_hidden_layers_lst[x] > 1):
        nn_sequential_permutation_model_npa[x].add(Dense(units=num_hidden_nodes_layer2_lst[x], activation=activation_function_lst[num_activation_functions_idx_lst[x]]))

    # If exist in this model permutation, add the third hidden layer
    if (num_hidden_layers_lst[x] > 2):
        nn_sequential_permutation_model_npa[x].add(Dense(units=num_hidden_nodes_layer3_lst[x], activation=activation_function_lst[num_activation_functions_idx_lst[x]]))         

    # Add the output layer to the model specifying the number of output neurons and activation function
    nn_sequential_permutation_model_npa[x].add(Dense(units=number_output_neurons, activation=activation_function_lst[num_activation_functions_idx_lst[x]]))

    # Display the Sequential model summary
    #nn_sequential_permutation_model_npa[x].summary()

    # Compile the Sequential model
    nn_sequential_permutation_model_npa[x].compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    # Fit the model and the training data
    nn_sequential_original_model_fit = nn_sequential_permutation_model_npa[x].fit(X_train_scaled_feature_drop_df, y_train_srs, epochs=num_epochs_lst[x], verbose=model_verbose) 

    # Evaluate the model loss and accuracy metrics using the evaluate method and the test data
    nn_sequential_permutation_model_loss, nn_sequential_permutation_model_accuracy =  nn_sequential_permutation_model_npa[x].evaluate(X_test_scaled_feature_drop_df,y_test_srs,verbose=0)

    # If this model permutation provided the best results, update "best" value set
    # (Finding best accuracy)
    if (nn_sequential_permutation_model_accuracy > best_accuracy_value):
        best_permutation_num = x
        best_loss_value = nn_sequential_permutation_model_loss
        best_accuracy_value = nn_sequential_permutation_model_accuracy
    
    # Display the model loss and accuracy results
    if (print_debug):
        print(f"\t\t\t***                       Permutation:{x+1} ==> Loss: {nn_sequential_permutation_model_loss}, Accuracy: {nn_sequential_permutation_model_accuracy} ***") 
        print(f"\t\t\t*** Best Running Results: Permutation:{best_permutation_num+1} ==> Loss: {best_loss_value}, Accuracy: {best_accuracy_value} ***")                 

    nn_sequential_permutation_model_fit_lst.append(nn_sequential_original_model_fit)
    nn_sequential_permutation_model_loss_lst.append(nn_sequential_permutation_model_loss)
    nn_sequential_permutation_model_accuracy_lst.append(nn_sequential_permutation_model_accuracy)
        
print(f"\n\n*** Final Best Results: Permutation:{best_permutation_num+1} ==> Loss: {best_loss_value}, Accuracy: {best_accuracy_value} ***")      


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Step 2: After finishing your models, display the accuracy scores achieved by each model, and compare the results.

# In[ ]:


# Plot the loss/accuracy of the original neural network model results
model_plot = pd.DataFrame(
    nn_sequential_original_model_fit.history, 
    index=range(1, len(nn_sequential_original_model_fit.history["loss"]) + 1)
)

# Display the model loss and accuracy results

print()
print(f"\n\n*** Original Neural Network Results: ==> Loss: {nn_sequential_permutation_model_loss}, Accuracy: {nn_sequential_permutation_model_accuracy} ***")      
print()

# Vizualize the model plot where the y-axis displays the loss metric
model_plot.plot(y="loss", title="History Of Original Neural Network Model: Loss", xlabel="# Epoch", ylabel="Loss")

# Vizualize the model plot where the y-axis displays the accuracy metric
model_plot.plot(y="accuracy", title="History Of Original Neural Network Model: Accuracy", xlabel="# Epoch", ylabel="Accuracy")


# In[ ]:


# Plot the loss/accuracy of the Best neural network model results that was obtained from all the different model permutation runs
model_plot = pd.DataFrame(
    nn_sequential_permutation_model_fit_lst[best_permutation_num].history, 
    index=range(1, len(nn_sequential_permutation_model_fit_lst[best_permutation_num].history["loss"]) + 1)
)

print()
print(f"\n\n*** Final Best Results: Permutation:{best_permutation_num+1} ==> Loss: {best_loss_value}, Accuracy: {best_accuracy_value} ***")      
print()

# Vizualize the model plot where the y-axis displays the loss metric
model_plot.plot(y="loss", title="History Of The Best Neural Network Model: Loss", xlabel="# Epoch", ylabel="Loss")

# Vizualize the model plot where the y-axis displays the accuracy metric
model_plot.plot(y="accuracy", title="History Of The Best Neural Network Model: Accuracy", xlabel="# Epoch", ylabel="Accuracy")


# ### Step 3: Save each of your alternative models as an HDF5 file.
# 

# In[ ]:


# Save the loss/accuracy of the original neural network model results

# Save model as JSON
nn_sequential_permutation_model_json = nn_sequential_original_model.to_json()

file_path = Path("./Resources/nn_sequential_original_model.json")
with open(file_path, "w") as json_file:
    json_file.write(nn_sequential_permutation_model_json)    
    
# Save weights
file_path = Path("./Resources/nn_sequential_original_model.h5")
nn_sequential_original_model.save_weights(file_path)    


# In[ ]:


# Save the loss/accuracy of the Best neural network model results that was obtained from all the different model permutation runs
nn_sequential_best_model_json = nn_sequential_permutation_model_npa[best_permutation_num].to_json()

file_path = Path("./Resources/nn_sequential_best_model_json.json")
with open(file_path, "w") as json_file:
    json_file.write(nn_sequential_best_model_json)    
    
# Save weights
file_path = Path("./Resources/nn_sequential_best_model_json.h5")
nn_sequential_permutation_model_npa[best_permutation_num].save_weights(file_path) 


# In[ ]:




