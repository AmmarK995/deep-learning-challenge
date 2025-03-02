# deep-learning-challenge

The purpose of this analysis was to develop a deep learning classification model to predict whether applicants for funding would be successful or unsuccessful based on various organizational and financial characteristics. The analysis was conducted with a csv dataset containing more than 34,000 organizations that have received funding from the nonprofit foundation Alphabet Soup over the years.

The first attempt was conducted in the Starter Code Jupyeter Notebook. The csv data was loaded from the 'https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv' URL in Google Colab. The first step towards building the deep learning model was to preprocess the data. To do this, the 'EIN' and 'NAME' columns were dropped, since their data was for identification purposes only and has no impact on predicting performance. The nature of the data was explored using functions such as nunique() and value_counts(). And subsequently once the applicable cut-off values for the classification lists and application types were defined, the next step was to convert categorical data to numeric with the `pd.get_dummies` function.

Next the x and y variables were defined. The target variable y was defined with the 'IS_SUCCESSFUL' category, essentially giving it a binary classification of whether or not the model is successful. The feature variable x was denoted by the remaining categorical and numerical variables in the dataset.

The train_test_split function was then used to split the preprocessed date into our features and target arrays (80% training and 20% testing). The StandardScaler tool from sklearn.preprocessing was used to normalize numerical variables for better neural network performance. The first model was compiled with the following features:

First hidden layer: 80
Second hidden layer: 30
Output layer: 1
Loss function: binary_crossentropy
Optimizer: adam
Batch size: 32
Epochs: 100

Two hidden layers were initially chosen for the first attempt as a basic starting point and to gauge the performance with the current data. ReLU activation helps with feature learning, while Sigmoid in the output layer ensures classification probabilities. Binary Crossentropy loss is suitable for binary classification problems.

The first attempt yielded a test accuracy of 72.67% and a Test LOss of 0.5687.

Though not ideal, this result is not entirely terrible. Regardless, multiple attempts were made in the AlphabetSoupCharity_Optimization Jupyter Notebook in the Optimization folder to improve the accuracy.

## Optimization 

For the first attempt in the optimization journey, the neurons were increased for each layer:

First Hidden Layer:  80 --> 100
Second Hidden Layer: 30 --> 50

This was done because increasing eurons allows the model to capture more complex relationships within the data, and therefore may help improve the accuracy.

The Epochs were also increased from: 100 --> 150

This was done because training longer also helps the model converge to a better accuracy level.

The above steps resulted in an Accuracy of 72.76% and Loss of 0.5787.

Though this was a slight improvement, the gain in accuracy is rather minimal. Therefore a second attemp was made:

First Hidden Layer:  100 --> 128
Second Hidden Layer: 50 --> 64
Third Hidden Layer: 25

The neurons were increased again and a Third Hidden Layer was added to help enhance deep feature extraction, and potentially improve accuracy.

The Epochs were also increased from: 150 --> 200

However the results from the second attempt gave an accuracy of 72.71% and Test Loss of 0.6055. 

Since the effort was actually counterproductive, for the third attempt, some columns were dropped to remove unnecessary data.

The corr()["IS_SUCCESSFUL"].sort_values function was used to check the correlation of numerical features with the target variable. Subsequently the drop function was used to drop nine columns that had the closest correlations to 0.

The Neurons for the Third Hidden Layer were also increased: 25 --> 32
Epochs were also increased for the final attempt: 200 --> 250

However, this model resulted in an Accuracy of 72.30% and Loss of 0.6136.

In light of the diminishing returns, no further testing was done. It may be possible to improve accuracy using a trial/error method with removing/adding columns, but seems unlikely. Due to the nature of the data, it may be more fruitful to explore alternative approaches, such as Random Forest or XGBoost. Such other ML models may be able to provide better accuracy for this dataset.

The final results are summarized below:

## Baseline Model
Test Accuracy: 72.67%
Test Loss: 0.5687

## Optimization - First Attempt (Best result)
Test Accuracy: 72.76%
Test Loss: 0.5787

## Optimization - Third Attempt
Test Accuracy: 72.30%
Test Loss: 0.6136

As we can see, the third attempt at optimization yielded the worst result so far. Therefore, it makes no sense to continue optimization. Exploring alternative models such as Random Forest or XGBoost may yield better results.
