# Machine Learning

## Machine Learning Algorithms

<!-- index -->
* [Linear Regression](#linear-regression) <!-- done -->
* [Logistic Regression](#logistic-regression)
* [Decision Tree](#decision-tree) <!-- done -->
* [Random Forest](#random-forest)
* [Support Vector Machine](#support-vector-machine)
* [K-Nearest Neighbors](#k-nearest-neighbors) <!-- done -->
* [K-Means](#k-means)
* [Naive Bayes](#naive-bayes)
* [Principal Component Analysis](#principal-component-analysis)
* [Gradient Boosting](#gradient-boosting)
* [Reinforcement Learning](#reinforcement-learning)
* [Dimensionality Reduction](#dimensionality-reduction)
* [Model Selection](#model-selection)
* [Ensemble Learning](#ensemble-learning)
* [Bias-Variance Tradeoff](#bias-variance-tradeoff)
* [Regularization](#regularization)
* [Evaluation Metrics](#evaluation-metrics)
* [Feature Engineering](#feature-engineering)
* [Feature Selection](#feature-selection)
* [Cross-Validation](#cross-validation) 
* [Bagging](#bagging)
* [Boosting](#boosting)
* [Gradient Descent](#gradient-descent)

<!-- indexstop -->
<hr>

### Linear Regression

* [Linear Regression](https://www.youtube.com/watch?v=ZkjP5RJLQF4&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF&index=3) - StatQuest with Josh Starmer
1. **Definition:** Linear regression is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables).

2. Linear regression is a linear model, i.e., a model that assumes a linear relationship between the input variables (x) and the single output variable (y). More specifically, y can be calculated from a linear combination of the input variables (x).

<hr>

### Decision Tree
1. **Definition:** A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility.

A decision tree is a predictive modeling tool that maps observations about an item to conclusions about the item's target value. It is a tree-like model where an internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents the predicted target value.

2. **Description:** Decision trees are a non-linear model used for both classification and regression tasks. They partition the data recursively based on feature values to make decisions.

3. **Type:** Supervised Learning.

4. **Reason to use**: Decision trees are easy to interpret and visualize, require little data preparation, and can handle both numerical and categorical data. They are also non-parametric, which means that they make no assumptions about the distribution of the variables in each class.

5. **Difference between classification and regression trees :** Classification trees are used when the target variable is categorical, whereas regression trees are used when the target variable is continuous.

6. **Difference between decision trees and neural network :**

| Decision Trees | Neural Networks |
| --- | --- |
| Decision trees are non-parametric models. | Neural networks are parametric models. |
| Decision trees are easy to interpret and visualize. | Neural networks are difficult to interpret and visualize. |
| Decision trees are computationally less expensive. | Neural networks are computationally expensive. |
| Decision trees are prone to overfitting. | Neural networks are prone to overfitting. |
| Decision trees are less flexible. | Neural networks are more flexible. |
| Decision trees are less accurate. | Neural networks are more accurate. |

7. **Pros :**
   - Easy to interpret and visualize.
   - It require little data preparation.
   - It can handle both numerical and categorical data.
   - Non-parametric, which means that they make no assumptions about the distribution of the variables in each class.
   - Computationally less expensive.
   - Prone to overfitting.
   - Less flexible.
   - Less accurate.

8. **Cons :**
   - Prone to overfitting.
   - Less flexible.
   - Less accurate.
   - Small variations in the data can result in different trees, leading to instability.
   - In classification problems with imbalanced classes, decision trees may be biased toward the dominant class.

9. **How it works :** Decision trees are a non-linear model used for both classification and regression tasks. They partition the data recursively based on feature values to make decisions.

10. **How to choose the best split :** The best split is the one that results in the most homogeneous sub-nodes. The homogeneity of a node is measured by the impurity of its child nodes. The impurity of a node is calculated using an impurity function, e.g., Gini impurity or entropy. The split with the lowest impurity is selected as the best split. The process is repeated recursively until all the nodes are homogeneous. 

11. **Explanation in layman's terms :** Imagine you have a fruit basket with different fruits. To identify each fruit, you start with a question like "Is it round?" If yes, you ask another question like "Is it red?" If no, you might ask "Is it yellow?" This process continues until you reach a conclusion, like "It's an apple." Each question and answer form a branch in the decision tree, helping you classify the fruit.

12. **How probability is used :** Probability is often used to estimate the likelihood of different outcomes at each node. The algorithm makes decisions by evaluating features and splitting data into subsets based on certain conditions. At each decision node, probabilities are assigned to potential outcomes, and the algorithm chooses the path with the highest probability. 
    - For example, in classification tasks, decision trees may assign probabilities to different classes for a given set of features. The path with the highest probability becomes the predicted class. In regression tasks, decision trees might estimate the probability distribution of the target variable at each leaf node.

    - Probability helps decision trees make informed choices at each step, leading to a structured decision-making process in machine learning models.

13. **Explanation of usage of Probability in layman's term :** Imagine decision trees as a flowchart for making decisions. Each step in the flowchart is a question about something, like "Is it raining?" or "Is it sunny?" The answers to these questions lead you down different paths.

    - Now, instead of just saying "yes" or "no" at each step, decision trees use probability to say how likely something is. For example, if the question is "Will it rain tomorrow?" the tree might say there's a 70% chance of rain and a 30% chance of no rain.

    - So, decision trees help us make choices by considering the likelihood of different outcomes at each step. It's like playing a game and deciding the best move based on the chances of winning or losing.

<hr>

### Random Forest


1. **Definition :** A Random Forest is an ensemble learning method that builds a collection of decision trees during training and outputs the mode (classification) or average prediction (regression) of the individual trees. It's called a "forest" because it consists of a multitude of trees.

2. **Ensemble Learning :** Ensemble learning involves combining multiple models to create a stronger and more robust model. In the case of Random Forest, the individual models are decision trees.

3. **Construction Process:**
   - **Bootstrap Sampling (Bagging):** Random Forest starts by creating multiple subsets of the original dataset through a process called bootstrap sampling. This involves randomly selecting samples with replacement, allowing some instances to appear in the subset multiple times, while others may not appear at all.

   - **Tree Construction:** For each subset, a decision tree is constructed. However, during the construction of each tree, only a random subset of features is considered at each split. This introduces diversity among the trees.

   - **Voting or Averaging:** Once all the trees are built, they "vote" on the predicted outcome for classification tasks or contribute their predictions for regression tasks. The final prediction is determined by the majority vote (for classification) or the average (for regression).

4. **Type :** Random Forest is an ensemble learning technique that falls under supervised learning. It is versatile and can be applied to both classification and regression problems.

5. **Reasons to Use Random Forest:**
   - **High Accuracy:** Random Forest often achieves high accuracy by reducing overfitting and capturing diverse patterns in the data.
   - **Robustness:** The ensemble approach makes Random Forest robust to noisy data and outliers.
   - **Feature Importance:** It provides insights into feature importance, helping identify which features are most influential in making predictions.

6. **Pros:**
   - **Reduces Overfitting:** By combining multiple trees, Random Forest mitigates the risk of overfitting present in individual decision trees.
   - **Handles Large Datasets:** It performs well on large datasets with high dimensionality.
   - **Works for Both Classification and Regression:** Versatility in addressing different types of predictive tasks.

7. **Cons:**
   - **Less Interpretability:** Random Forests are less interpretable compared to individual decision trees.
   - **Computational Complexity:** Training multiple trees can be computationally expensive.
   - **Potential for Overfitting:** While it reduces overfitting, in certain cases with noisy data, it may still overfit the training data.

8. **Improvements Over Individual Decision Trees :** Random Forest improves upon individual decision trees by introducing randomness in the construction process. This randomness comes from using subsets of the data and subsets of features at each split, leading to a more robust and accurate model.

9. **Significance of the Ensemble Approach :** The significance of the ensemble approach lies in the diversity of the individual trees. Since each tree is trained on a different subset of data and features, they capture different aspects of the underlying patterns in the data. Combining these diverse perspectives results in a more reliable and generalized model.

10. **Use of Probability in Random Forest :** Random Forest indirectly uses probability through the voting mechanism. 

      - In classification, the final prediction is based on the majority vote of the trees. Each tree "votes" for a particular class, and the class with the most votes is considered the final prediction. This voting process inherently involves a probabilistic element.

      - In summary, Random Forest is a powerful ensemble learning method that leverages decision trees to provide accurate and robust predictions. Its ability to reduce overfitting, handle large datasets, and offer insights into feature importance makes it widely used in various machine learning applications.


<hr>

### k-Nearest Neighbors
1. **Definition:** k-Nearest Neighbors is a supervised machine learning algorithm for classification or regression.

2. The k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression:
   - In k-NN classification, the output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
   - In k-NN regression, the output is the property value for the object. This value is the average of the values of its k nearest neighbors.

4. **Type:** Supervised Learning.

5. **Reason to use:** The k-nearest neighbors algorithm is a robust classifier that is often used as a benchmark for more complex classifiers such as artificial neural networks (ANN) and support vector machines (SVM). Despite its simplicity, it can outperform more powerful classifiers and is used in a variety of applications such as economic forecasting, data compression, and genetics.

6. **How it works:** The k-nearest neighbors algorithm is based on the simple idea of predicting unknown values by matching them with the most similar known values. The algorithm creates a model by finding the k most similar records to a given record and then making an estimate based on the known output values of those records. The similarity of two records is determined using a distance function.

7. **Pros:**
   - The algorithm is simple and easy to implement.
   - There’s no need to build a model, tune several parameters, or make additional assumptions.
   - The algorithm is versatile; it can be used for classification, regression, and search.
   - Useful for low-dimensional feature spaces.
   - It's a good baseline model to test your classification performance.
   - The algorithm is insensitive to outliers, irrelevant features, and the scale of the data.

8. **Cons:**
   - The algorithm gets significantly slower as the number of examples and/or predictor variables increase.
   - The algorithm doesn’t learn anything from the training data; it simply uses the training data at the time of prediction.
   - The algorithm is sensitive to irrelevant features and the scale of the data.

9. **How to choose k:** The best choice of k depends on the data; generally, larger values of k reduce the effect of noise on the classification but make boundaries between classes less distinct. A good k can be selected by various heuristic techniques, for example, cross-validation. The special case where the class is predicted to be the class of the closest training sample (i.e., when k = 1) is called the nearest neighbor algorithm.

10. **How to choose the distance function:** The distance function can be any metric measure; standard Euclidean distance is the most common choice. In that case, the class of a given point is determined by the majority of votes from its k nearest neighbors. Other popular metrics include the Manhattan distance, Euclidean distance, Hamming distance, Minkowski distance, cosine distance, etc.

11. **How to choose the optimal number of neighbors:** The optimal choice of the value k is highly data-dependent: in general, a larger k suppresses the effects of noise but makes the classification boundaries less distinct. A good k can be selected by various heuristic techniques, e.g., cross-validation. The special case where the class is predicted to be the class of the closest training sample (i.e., when k = 1) is called the nearest neighbor algorithm.

12. **Explanation in layman's terms:** The k-nearest neighbors algorithm is based on the simple idea of predicting unknown values by matching them with the most similar known values. The algorithm creates a model by finding the k most similar records to a given record and then making an estimate based on the known output values of those records. The similarity of two records is determined using a distance function.

13. **Type of model or architecture:** The k-nearest neighbors algorithm is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression.