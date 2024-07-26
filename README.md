# Unsupervised Learning Exercises

## Introduction to Unsupervised Learning

- Types of Machine Learning Problems
  - **Supervised Learning**: Learning from labeled data to predict outcomes (e.g., classification and regression).
  - **Unsupervised Learning**: Learning from unlabeled data to find patterns or structure (e.g., clustering and dimensionality reduction).
  - **Reinforcement Learning**: Learning through feedback to maximize a cumulative reward (e.g., game playing and robotics).

- Importance of Unsupervised Learning
  - **Data Availability**: Large amounts of unlabeled data are often available, while labeled data can be expensive and time-consuming to obtain.
  - **Discovering Hidden Patterns**: Helps in finding patterns or structures in data that are not immediately obvious.
  - **Preprocessing**: Can be used for preprocessing steps such as dimensionality reduction before applying supervised learning methods.

- Goals of Unsupervised Learning
  - **Dimensionality Reduction**: Reduce the number of variables under consideration, making the data easier to visualize and analyze.
  - **Clustering**: Group similar data points together to discover subgroups within the data.
  - **Data Visualization**: Create informative visual representations of the data to aid in understanding.
  - **Data Preprocessing**: Prepare data for other tasks, such as supervised learning.
  - **Feature Learning**: Automatically discover the representations needed for feature detection or classification.
  - **Anomaly Detection**: Identify rare items, events, or observations which raise suspicions by differing significantly from the majority of the data.

## Dimensionality Reduction

- Introduction to Dimensionality Reduction
  - **Definition**: The process of reducing the number of random variables under consideration by obtaining a set of principal variables.
  - **Purpose**: Simplify models, reduce computational cost, and mitigate the curse of dimensionality.
  - **Techniques**: Various methods such as PCA, t-SNE, and LDA.

- Principal Component Analysis (PCA)
  - **PCA Methodology**
    - **Covariance Matrix**: Calculate the covariance matrix of the data.
    - **Eigenvectors and Eigenvalues**: Determine the eigenvectors and eigenvalues of the covariance matrix.
    - **Principal Components**: The eigenvectors represent the principal components.
    - **Projection**: Project the data onto the principal components.
  - **Choosing Principal Components**
    - **Variance Explanation**: Select components that explain the maximum variance in the data.
    - **Elbow Method**: Use the elbow method to determine the optimal number of components.
    - **Scree Plot**: Visualize the explained variance to make an informed decision.
  - **Applications of PCA**
    - **Data Visualization**: Reduce dimensions for visualization in 2D or 3D.
    - **Noise Reduction**: Remove noise by discarding components with low variance.
    - **Feature Extraction**: Derive new features that summarize the original features.
  - **Principal Component Regression (PCR)**
    - **Methodology**: Use principal components as predictors in a regression model.
    - **Advantages**: Addresses multicollinearity by using uncorrelated components.
    - **Challenges**: Interpretation of principal components can be difficult.
  - **Face Recognition with PCA**
    - **Eigenfaces**: Represent faces as a combination of principal components called eigenfaces.
    - **Recognition Process**: Project new face images onto the eigenface space and compare with stored projections.
    - **Advantages**: Efficient representation and recognition of faces.

## Clustering
- Introduction to Clustering
  - **Definition**: The task of grouping a set of objects in such a way that objects in the same group (cluster) are more similar to each other than to those in other groups.
  - **Purpose**: Discover the underlying structure of the data, identify patterns, and provide insights.

- Applications of Clustering
  - **Marketing and Sales**
    - **Customer Segmentation**: Identify subgroups of customers with similar behaviors for targeted marketing.
    - **Product Recommendation**: Group similar products for recommendation systems.
  - **Disease Subtypes Discovery**
    - **Healthcare**: Identify different subtypes of a disease to tailor treatments.
    - **Genomic Research**: Cluster genes with similar expression patterns.
  - **Document Clustering**
    - **Information Retrieval**: Group similar documents to improve search and retrieval efficiency.
    - **Topic Modeling**: Identify topics within a large collection of documents.
  - **Image Segmentation**
    - **Computer Vision**: Divide an image into segments for easier analysis and processing.
    - **Medical Imaging**: Identify and isolate regions of interest in medical scans.

- Popular Clustering Methods
  - **K-means Clustering**
    - **K-means Algorithm**
      - **Initialization**: Select initial cluster centroids.
      - **Assignment**: Assign each data point to the nearest centroid.
      - **Update**: Recalculate the centroids based on the assigned points.
      - **Iteration**: Repeat assignment and update steps until convergence.
    - **Choosing the Number of Clusters**
      - **Elbow Method**: Plot the explained variance as a function of the number of clusters and look for the "elbow".
      - **Silhouette Score**: Measure the quality of the clustering to determine the optimal number of clusters.
  - **Hierarchical Clustering**
    - **Hierarchical Clustering Algorithm**
      - **Agglomerative Method**: Start with each point as a single cluster and merge the closest pairs iteratively.
      - **Divisive Method**: Start with all points in one cluster and recursively split them.
    - **Linkage Types and Metrics**
      - **Complete Linkage**: Maximum distance between points in clusters.
      - **Single Linkage**: Minimum distance between points in clusters.
      - **Average Linkage**: Average distance between points in clusters.
      - **Ward's Method**: Minimize the variance of the clusters being merged.
    - **Dendrograms and Cluster Identification**
      - **Dendrogram**: A tree-like diagram that records the sequences of merges or splits.
      - **Cluster Identification**: Cut the dendrogram at the desired level to form clusters.

- Importance of Similarity Metrics and Feature Scaling
  - **Similarity Metrics**
    - **Euclidean Distance**: Straight-line distance between two points.
    - **Manhattan Distance**: Sum of the absolute differences of the coordinates.
    - **Cosine Similarity**: Measure of cosine of the angle between two vectors.
    - **Jaccard Index**: Measure of similarity between two sets.
  - **Feature Scaling**
    - **Standardization**: Transform data to have zero mean and unit variance.
    - **Normalization**: Scale data to a fixed range, typically [0, 1].
    - **Importance**: Ensures that features contribute equally to the distance calculations and improves the performance of clustering algorithms.

## Recommender Systems

- Introduction to Recommender Systems
  - **Definition**: Systems that provide personalized recommendations to users based on their preferences and behavior.
  - **Purpose**: Enhance user experience by suggesting relevant items, improve user engagement, and increase sales or content consumption.

- Popular Approaches
  - **Content-Based Filtering**
    - **Definition**: Recommends items similar to those a user has liked in the past based on item attributes.
    - **Method**: Create user and item profiles, and recommend items with attributes similar to those the user has shown interest in.
    - **Advantages**: Does not require data from other users, handles new items well.
    - **Disadvantages**: Limited by the quality and completeness of item attributes, does not handle serendipity well.
  - **Collaborative Filtering**
    - **Memory-Based Approaches**
      - **User-User Collaborative Filtering**: Recommend items liked by similar users.
      - **Item-Item Collaborative Filtering**: Recommend items similar to those the user has liked.
      - **Advantages**: Simple to implement, often very effective.
      - **Disadvantages**: Struggles with new users (cold start problem) and sparsity of user-item interactions.
    - **Latent Factor Modeling**
      - **Definition**: Use matrix factorization techniques to uncover latent factors representing user and item characteristics.
      - **Methods**: Singular Value Decomposition (SVD), Non-negative Matrix Factorization (NMF).
      - **Advantages**: Can capture complex patterns in user-item interactions, often more accurate.
      - **Disadvantages**: Computationally intensive, requires a large amount of data.

- Utility Matrix and Similarity Measures
  - **Utility Matrix**
    - **Explicit Ratings**: Users provide ratings for items (e.g., 1-5 stars).
    - **Implicit Ratings**: Inferred from user behavior (e.g., clicks, purchases).
  - **Cosine Similarity**
    - **Definition**: Measures the cosine of the angle between two vectors in a multi-dimensional space.
    - **Use Case**: Often used in item-item and user-user collaborative filtering.
  - **Jaccard Similarity**
    - **Definition**: Measures the similarity between two sets by comparing the size of the intersection to the size of the union.
    - **Use Case**: Useful for binary data (e.g., implicit feedback).
  - **Distance-Based Similarity**
    - **Manhattan Distance**: Sum of absolute differences between points.
    - **Euclidean Distance**: Straight-line distance between points.
    - **Minkowski Distance**: Generalization of Euclidean and Manhattan distances.
  - **Pearson Correlation**
    - **Definition**: Measures linear correlation between two variables.
    - **Use Case**: Used to find similarity between users or items based on their ratings.

- Challenges and Considerations in Large-Scale Recommender Systems
  - **Scalability**: Efficiently handling large numbers of users and items.
    - **Solutions**: Use of approximation algorithms, parallel processing, and distributed computing.
  - **Sparsity**: Dealing with the large amount of missing data in the utility matrix.
    - **Solutions**: Imputation methods, leveraging implicit feedback, matrix factorization techniques.
  - **Cold Start Problem**: Difficulty in recommending items to new users or recommending new items.
    - **Solutions**: Use hybrid approaches, incorporate side information (e.g., user demographics, item attributes).
  - **Diversity and Serendipity**: Ensuring recommendations are not only relevant but also diverse and occasionally surprising.
    - **Solutions**: Diversification algorithms, balancing exploration and exploitation.
  - **Evaluation**: Measuring the effectiveness of recommender systems.
    - **Metrics**: Precision, recall, F1 score, mean average precision (MAP), root mean square error (RMSE).
    - **Methods**: A/B testing, cross-validation, offline and online evaluation.

## Matrix Factorization

- Introduction to Matrix Factorization
  - **Definition**: Matrix factorization techniques decompose a matrix into product matrices to uncover latent factors.
  - **Purpose**: Reduce dimensionality, uncover hidden patterns, and improve computational efficiency in various applications.

- Applications of Matrix Factorization
  - **Audio Signal Separation**
    - **Definition**: Decomposing mixed audio signals into individual sources.
    - **Example**: Separating vocals from background music.
  - **Analytic Chemistry**
    - **Definition**: Analyzing chemical mixtures by separating component spectra.
    - **Example**: Identifying substances in a sample using spectroscopic data.
  - **Gene Expression Analysis**
    - **Definition**: Identifying patterns in gene expression data.
    - **Example**: Discovering gene regulatory networks.
  - **Recommender Systems**
    - **Definition**: Predicting user preferences by decomposing the user-item interaction matrix.
    - **Example**: Netflix's recommendation algorithm.

- Methods of Matrix Factorization
  - **Singular Value Decomposition (SVD)**
    - **Definition**: Decomposes a matrix into three matrices: \(U\), \(\Sigma\), and \(V^T\).
    - **Properties**: Captures the most significant singular values, useful for noise reduction.
    - **Application**: Used in Latent Semantic Analysis (LSA) for text data.
  - **Non-negative Matrix Factorization (NMF)**
    - **Definition**: Decomposes a matrix into two non-negative matrices \(W\) and \(H\).
    - **Properties**: Suitable for data with non-negative values, interpretable as parts-based representation.
    - **Application**: Used in image and audio processing, bioinformatics.

- Loss Functions in NMF
  - **L2 Loss**
    - **Definition**: Measures the sum of squared differences between the original and approximated matrices.
    - **Application**: General purpose, sensitive to outliers.
  - **L1 Loss**
    - **Definition**: Measures the sum of absolute differences between the original and approximated matrices.
    - **Application**: Robust to outliers, promotes sparsity.
  - **KL Loss**
    - **Definition**: Kullback-Leibler divergence measures the difference between probability distributions.
    - **Application**: Suitable for probabilistic data, interpretable as information loss.
  - **Itakura-Saito Loss**
    - **Definition**: Measures the multiplicative difference between the original and approximated matrices.
    - **Application**: Used in audio signal processing for its perceptual relevance.

- Optimization in NMF
  - **Gradient Descent**
    - **Definition**: Iteratively updates the factor matrices \(W\) and \(H\) to minimize the loss function.
    - **Method**: Compute gradients of the loss function with respect to \(W\) and \(H\), then update them using a learning rate.
    - **Challenges**: Convergence to local minima, choice of learning rate.
  - **Practical Usage of NMF**
    - **Libraries**: Available in machine learning libraries such as scikit-learn.
    - **Implementation**: Choose appropriate initialization, set regularization parameters, and determine the number of latent factors.
    - **Example**: `sklearn.decomposition.NMF` for applying NMF to datasets.

