# Machine Learning

## Machine Learning Algorithms
<!-- index -->
* [Linear Regression](#linear-regression) <!-- done -->
* [Logistic Regression](#logistic-regression)
* [Decision Tree](#decision-tree)
* [Random Forest](#random-forest)
* [Support Vector Machine](#support-vector-machine)
* [K-Nearest Neighbors](#k-nearest-neighbors)
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
* [Cross Validation](#cross-validation)
* [Bagging](#bagging)
* [Boosting](#boosting)
* [Gradient Descent](#gradient-descent)

<!-- indexstop -->

### Linear Regression
* [Linear Regression](https://www.youtube.com/watch?v=ZkjP5RJLQF4&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF&index=3) - StatQuest with Josh Starmer
1. Definition: Linear regression is a linear approach to modeling the relationship between a scalar response `(or dependent variable)` and one or more explanatory variables `(or independent variables)`.
2. Linear regression is a linear model, e.g. a model that assumes a linear relationship between the input variables `(x)` and the single output variable `(y)`. More specifically, that `y` can be calculated from a linear combination of the input variables `(x)`.

### k-Nearest Neighbors
1. `Definition`: k-Nearest Neighbors is a supervised machine learning algorithm for classification or regression.

2. The k-nearest neighbors algorithm (k-NN) is a `non-parametric method` used for classification and regression. In both cases, the input consists of the `k` closest training examples in the feature space. The output depends on whether `k-NN` is used for classification or regression:
    * In `k-NN` classification, the output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its `k` nearest neighbors (`k` is a positive integer, typically small). If `k = 1`, then the object is simply assigned to the class of that single nearest neighbor.
    * In `k-NN` regression, the output is the property value for the object. This value is the average of the values of its `k` nearest neighbors.
4. Type: `Supervised Learning`.

5. `Reason to use`: The k-nearest neighbors algorithm is a robust classifier that is often used as a benchmark for more complex classifiers such as artificial neural networks (ANN) and support vector machines (SVM). Despite its simplicity, it can outperform more powerful classifiers and is used in a variety of applications such as `economic forecasting, data compression and genetics`.

6. `How it works`: The k-nearest neighbors algorithm is based around the simple idea of predicting unknown values by matching them with the most similar known values. The algorithm creates a model by `finding the k` most similar records to a given record and then making an estimate based on the known output values of those records. The similarity of two records is determined using a distance function.

7. `Pros`: The algorithm is simple and easy to implement. There’s no need to build a model, tune several parameters, or make additional assumptions. The algorithm is versatile. It can be used for classification, regression, and search (as we will see in the next section). The algorithm is `useful for `low-dimensional` feature spaces`. It’s a good `baseline` model to `test` your `classification performance`. The algorithm is `insensitive to outliers`, irrelevant features, and the `scale of the data`.

8. `Cons`: The algorithm gets significantly `slower` as the `number of examples` and/or `predictor variables increase`. The algorithm doesn’t `learn` anything from the training data. It simply uses the training data at the time of prediction. The algorithm is `sensitive to irrelevant features` and the `scale of the data`.

9. `How to choose k`: The best choice of k depends upon the data; generally, larger values of k reduce the effect of noise on the classification, but make boundaries between classes less distinct. A good k can be selected by various heuristic techniques, for example, cross-validation. The special case where the class is predicted to be the class of the closest training sample (i.e. when k = 1) is called the nearest neighbor algorithm.

10. `How to choose distance function`: The distance function can be any `metric measure`: standard Euclidean distance is the most common choice. In that case, the `class of a given point is determined by the majority` of votes from its `k nearest neighbors`. Other `popular metrics` include the `Manhattan distance`, `Euclidean distance`, `Hamming distance`, `Minkowski distance`, `cosine distance` etc.

11. `How to choose the optimal number of neighbors`: The optimal choice of the value k is highly data-dependent: in general a larger k suppresses the effects of noise, but makes the classification boundaries less distinct. A good k can be selected by various heuristic techniques, e.g. cross-validation. The special case where the class is predicted to be the class of the closest training sample (i.e. when k = 1) is called the nearest neighbor algorithm.

12. `Explanation in layman terms`: The k-nearest neighbors algorithm is based around the simple idea of predicting unknown values by matching them with the most similar known values. The algorithm creates a model by finding the k most similar records to a given record and then making an estimate based on the known output values of those records. The similarity of two records is determined using a distance function.

13. `Which type of model or architecture it is`: The k-nearest neighbors algorithm is a `non-parametric` method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression.




