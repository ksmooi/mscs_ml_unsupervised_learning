# Building Recommender Systems for Movie Rating Prediction

In this assignment, we will build a recommender systems that predict movie ratings. [MovieLense](https://grouplens.org/datasets/movielens/) has currently 25 million user-movie ratings.  Since the entire data is too big, we use  a 1 million ratings subset [MovieLens 1M](https://www.kaggle.com/odedgolden/movielens-1m-dataset), and we reformatted the data to make it more convenient to use.


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import jaccard, cosine 
from pytest import approx
```


```python
MV_users = pd.read_csv('data/users.csv')
MV_movies = pd.read_csv('data/movies.csv')
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
```


```python
from collections import namedtuple
Data = namedtuple('Data', ['users','movies','train','test'])
data = Data(MV_users, MV_movies, train, test)
```

### Starter codes
Now, we will be building a recommender system which has various techniques to predict ratings. 
The `class RecSys` has baseline prediction methods (such as predicting everything to 3 or to average rating of each user) and other utility functions. `class ContentBased` and `class Collaborative` inherit `class RecSys` and further add methods calculating item-item similarity matrix. You will be completing those functions using what we learned about content-based filtering and collaborative filtering.

`RecSys`'s `rating_matrix` method converts the (user id, movie id, rating) triplet from the train data (train data's ratings are known) into a utility matrix for 6040 users and 3883 movies.    
Here, we create the utility matrix as a dense matrix (numpy.array) format for convenience. But in a real world data where hundreds of millions of users and items may exist, we won't be able to create the utility matrix in a dense matrix format (For those who are curious why, try measuring the dense matrix self.Mr using .nbytes()). In that case, we may use sparse matrix operations as much as possible and distributed file systems and distributed computing will be needed. Fortunately, our data is small enough to fit in a laptop/pc memory. Also, we will use numpy and scipy.sparse, which allow significantly faster calculations than calculating on pandas.DataFrame object.    
In the `rating_matrix` method, pay attention to the index mapping as user IDs and movie IDs are not the same as array index.


```python
from sklearn.metrics.pairwise import cosine_similarity

class RecSys():
    def __init__(self,data):
        self.data=data
        self.allusers = list(self.data.users['uID'])
        self.allmovies = list(self.data.movies['mID'])
        self.genres = list(self.data.movies.columns.drop(['mID', 'title', 'year']))
        self.mid2idx = dict(zip(self.data.movies.mID,list(range(len(self.data.movies)))))
        self.uid2idx = dict(zip(self.data.users.uID,list(range(len(self.data.users)))))
        self.Mr=self.rating_matrix()
        self.Mm=None 
        self.sim=np.zeros((len(self.allmovies),len(self.allmovies)))
        
    def rating_matrix(self):
        """
        Convert the rating matrix to numpy array of shape (#allusers,#allmovies)
        """
        ind_movie = [self.mid2idx[x] for x in self.data.train.mID] 
        ind_user = [self.uid2idx[x] for x in self.data.train.uID]
        rating_train = list(self.data.train.rating)
        
        return np.array(coo_matrix((rating_train, (ind_user, ind_movie)), shape=(len(self.allusers), len(self.allmovies))).toarray())

    def predict_everything_to_3(self):
        """
        Predict everything to 3 for the test data
        """
        # Generate an array with 3s against all entries in test dataset
        return np.full(len(self.data.test), 3)
                
    def predict_to_user_average(self):
        """
        Predict to average rating for the user.
        Returns numpy array of shape (#users,)
        """
        # Generate an array as follows:
        # 1. Calculate all avg user rating as sum of ratings of user across all movies/number of movies whose rating > 0
        # 2. Return the average rating of users in test data
        
        user_avg_ratings = np.zeros(len(self.allusers))
        
        for user_idx in range(len(self.allusers)):
            user_ratings = self.Mr[user_idx]
            rated_items = user_ratings > 0
            if np.any(rated_items):
                user_avg_ratings[user_idx] = user_ratings[rated_items].mean()
            else:
                user_avg_ratings[user_idx] = 3  # default to 3 if no ratings
        
        test_user_idxs = [self.uid2idx[uid] for uid in self.data.test.uID]
        return user_avg_ratings[test_user_idxs]
    
    def predict_from_sim(self,uid,mid):
        """
        Predict a user rating on a movie given userID and movieID
        """
        # Predict user rating as follows:
        # 1. Get entry of user id in rating matrix
        # 2. Get entry of movie id in sim matrix
        # 3. Employ 1 and 2 to predict user rating of the movie
        
        user_idx = self.uid2idx[uid]
        movie_idx = self.mid2idx[mid]
        user_ratings = self.Mr[user_idx]
        
        # Get similarities and ratings for movies that the user has rated
        rated_items = user_ratings > 0
        similarities = self.sim[movie_idx, rated_items]
        ratings = user_ratings[rated_items]
        
        if len(ratings) == 0:  # If user hasn't rated any movies, return default rating of 3
            return 3
        
        # Calculate the weighted average of the ratings based on similarities
        weighted_sum = np.dot(similarities, ratings)
        sim_sum = np.sum(similarities)
        
        if sim_sum == 0:  # If no similarities, return the average of user's ratings
            return ratings.mean()
        
        return weighted_sum / sim_sum

    def predict(self):
        """
        Predict ratings in the test data. Returns predicted rating in a numpy array of size (# of rows in testdata,)
        """                
        predictions = []
        for _, row in self.data.test.iterrows():
            uid = row['uID']
            mid = row['mID']
            pred = self.predict_from_sim(uid, mid)
            predictions.append(pred)
        return np.array(predictions)
    
    def rmse(self,yp):
        yp = np.array(yp, dtype=np.float64)  # Ensure yp is a numpy array of floats
        yp[np.isnan(yp)] = 3  # In case there are nan values in prediction, it will impute to 3.
        yt = np.array(self.data.test.rating, dtype=np.float64)
        return np.sqrt(((yt - yp) ** 2).mean())


class ContentBased(RecSys):
    def __init__(self,data):
        super().__init__(data)
        self.data=data
        self.Mm = self.calc_movie_feature_matrix()  
        
    def calc_movie_feature_matrix(self):
        """
        Create movie feature matrix in a numpy array of shape (#allmovies, #genres) 
        """
        # Extract genre columns from movies DataFrame and convert to numpy array
        movie_features = self.data.movies[self.genres].to_numpy()
        return movie_features
    
    def calc_item_item_similarity(self):
        """
        Create item-item similarity using Jaccard similarity
        """
        # Update the sim matrix by calculating item-item similarity using Jaccard similarity
        # Jaccard Similarity: J(A, B) = |A∩B| / |A∪B| 
        
        num_movies = self.Mm.shape[0]
        self.sim = np.zeros((num_movies, num_movies))
        
        for i in range(num_movies):
            for j in range(num_movies):
                if i != j:
                    self.sim[i, j] = 1 - jaccard(self.Mm[i], self.Mm[j])
                else:
                    self.sim[i, j] = 1  # Jaccard similarity with itself is 1
        

class Collaborative(RecSys):    
    def __init__(self,data):
        super().__init__(data)
        
    def calc_item_item_similarity(self, simfunction, *X):  
        """
        Create item-item similarity using similarity function. 
        X is an optional transformed matrix of Mr
        """    
        # General function that calculates item-item similarity based on the sim function and data inputed
        if len(X)==0:
            self.sim = simfunction()            
        else:
            self.sim = simfunction(X[0]) # *X passes in a tuple format of (X,), to X[0] will be the actual transformed matrix
    
    def cossim(self):
        """
        Calculates item-item similarity for all pairs of items using cosine similarity (values from 0 to 1) on utility matrix
        Returns a cosine similarity matrix of size (#all movies, #all movies)
        """
        # Return a sim matrix by calculating item-item similarity for all pairs of items using Jaccard similarity
        # Cosine Similarity: C(A, B) = (A.B) / (||A||.||B||) 

        # Step 1: Impute unrated entries with user's average rating and normalize by subtracting the user's mean
        user_mean = np.true_divide(self.Mr.sum(1), (self.Mr != 0).sum(1))
        user_mean[np.isnan(user_mean)] = 0  # Handle division by zero for users with no ratings
        X = np.where(self.Mr != 0, self.Mr, user_mean[:, np.newaxis])
        X = X - user_mean[:, np.newaxis]

        # Step 2: Calculate cosine similarity for all item-item pairs
        norm = np.linalg.norm(X, axis=0)
        norm[norm == 0] = 1  # To avoid division by zero
        X_normalized = X / norm

        sim_matrix = np.dot(X_normalized.T, X_normalized)

        # Step 3: Rescale the cosine similarity to be 0~1
        sim_matrix = (sim_matrix + 1) / 2
        np.fill_diagonal(sim_matrix, 1)  # Fill diagonal with 1 as self similarity is 1

        self.sim = sim_matrix  # Update the similarity matrix
        return sim_matrix
        
    def jacsim(self,Xr):
        """
        Calculates item-item similarity for all pairs of items using jaccard similarity (values from 0 to 1)
        Xr is the transformed rating matrix.
        """    
        # Return a sim matrix by calculating item-item similarity for all pairs of items using Jaccard similarity
        # Jaccard Similarity: J(A, B) = |A∩B| / |A∪B| 

        # Ensure Xr is a CSR matrix for efficient row-wise operations
        Xr_sparse = csr_matrix(Xr, dtype=bool).astype(np.float32)

        # Calculate the number of non-zero elements for each item (column)
        item_nonzero = Xr_sparse.sum(axis=0).A1

        # Calculate cosine similarity (equivalent to intersection for binary data)
        intersection = cosine_similarity(Xr_sparse.T, dense_output=True)

        # Calculate union
        union = item_nonzero[:, np.newaxis] + item_nonzero[np.newaxis, :] - intersection

        # Calculate Jaccard similarity
        sim_matrix = np.divide(intersection, union, out=np.zeros_like(intersection), where=union!=0)

        # Set diagonal to 1
        np.fill_diagonal(sim_matrix, 1)

        return sim_matrix
```

# Q1. Baseline models [15 pts]

### 1a. Complete the function `predict_everything_to_3` in the class `RecSys`  [5 pts]


```python
# Creating Sample test data
np.random.seed(42)
sample_train = train[:30000]
sample_test = test[:30000]


sample_MV_users = MV_users[(MV_users.uID.isin(sample_train.uID)) | (MV_users.uID.isin(sample_test.uID))]
sample_MV_movies = MV_movies[(MV_movies.mID.isin(sample_train.mID)) | (MV_movies.mID.isin(sample_test.mID))]


sample_data = Data(sample_MV_users, sample_MV_movies, sample_train, sample_test)
```


```python
# Sample tests predict_everything_to_3 in class RecSys

sample_rs = RecSys(sample_data)
sample_yp = sample_rs.predict_everything_to_3()
print(sample_rs.rmse(sample_yp))
assert sample_rs.rmse(sample_yp)==approx(1.2642784503423288, abs=1e-3), "Did you predict everything to 3 for the test data?"
```

    1.2642784503423288



```python
# Hidden tests predict_everything_to_3 in class RecSys
rs = RecSys(data)
yp = rs.predict_everything_to_3()
print(rs.rmse(yp))
```

    1.2585510334053043


### 1b. Complete the function predict_to_user_average in the class RecSys [10 pts]
Hint: Include rated items only when averaging


```python
# Sample tests predict_to_user_average in the class RecSys
sample_yp = sample_rs.predict_to_user_average()
print(sample_rs.rmse(sample_yp))
assert sample_rs.rmse(sample_yp)==approx(1.1429596846619763, abs=1e-3), "Check predict_to_user_average in the RecSys class. Did you predict to average rating for the user?" 
```

    1.1429596846619763



```python
# Hidden tests predict_to_user_average in the class RecSys
yp = rs.predict_to_user_average()
print(rs.rmse(yp))
```

    1.0352910334228647


# Q2. Content-Based model [25 pts]

### 2a. Complete the function calc_movie_feature_matrix in the class ContentBased [5 pts]


```python
cb = ContentBased(data)
```


```python
# tests calc_movie_feature_matrix in the class ContentBased 
assert(cb.Mm.shape==(3883, 18))
```

### 2b. Complete the function calc_item_item_similarity in the class ContentBased [10 pts]
This function updates `self.sim` and does not return a value.    
Some factors to think about:     
1. The movie feature matrix has binary elements. Which similarity metric should be used?
2. What is the computation complexity (time complexity) on similarity calcuation?      
Hint: You may use functions in the `scipy.spatial.distance` module on the dense matrix, but it is quite slow (think about the time complexity). If you want to speed up, you may try using functions in the `scipy.sparse` module. 


```python
cb.calc_item_item_similarity()
```


```python
# Sample tests calc_item_item_similarity in ContentBased class 

sample_cb = ContentBased(sample_data)
sample_cb.calc_item_item_similarity() 

# print(np.trace(sample_cb.sim))
# print(sample_cb.sim[10:13,10:13])
assert(sample_cb.sim.sum() > 0), "Check calc_item_item_similarity."
assert(np.trace(sample_cb.sim) == 3152), "Check calc_item_item_similarity. What do you think np.trace(cb.sim) should be?"


ans = np.array([[1, 0.25, 0.],[0.25, 1, 0.],[0., 0., 1]])
for pred, true in zip(sample_cb.sim[10:13, 10:13], ans):
    assert approx(pred, 0.01) == true, "Check calc_item_item_similarity. Look at cb.sim"
```


```python
# tests calc_item_item_similarity in ContentBased class 
```


```python
# additional tests for calc_item_item_similarity in ContentBased class 
```


```python
# additional tests for calc_item_item_similarity in ContentBased class
```


```python
# additional tests for calc_item_item_similarity in ContentBased class
```


```python
# additional tests for calc_item_item_similarity in ContentBased class
```

### 2c. Complete the function predict_from_sim in the class RecSys [5 pts]


```python
# for a, b in zip(sample_MV_users.uID, sample_MV_movies.mID):
#     print(a, b, sample_cb.predict_from_sim(a,b))

# Sample tests for predict_from_sim in RecSys class 
assert(sample_cb.predict_from_sim(245,276)==approx(2.5128205128205128,abs=1e-2)), "Check predict_from_sim. Look at how you predicted a user rating on a movie given UserID and movieID."
assert(sample_cb.predict_from_sim(2026,2436)==approx(2.785714285714286,abs=1e-2)), "Check predict_from_sim. Look at how you predicted a user rating on a movie given UserID and movieID."
```


```python
# tests for predict_from_sim in RecSys class 
```

### 2d. Complete the function predict in the class RecSys [5 pts]
After completing the predict method in the RecSys class, run the cell below to calculate rating prediction and RMSE. How much does the performance increase compared to the baseline results from above? 


```python
# Sample tests method predict in the RecSys class 

sample_yp = sample_cb.predict()
sample_rmse = sample_cb.rmse(sample_yp)
print(sample_rmse)

assert(sample_rmse==approx(1.1962537249116723, abs=1e-2)), "Check method predict in the RecSys class."
```

    1.1966092802947408



```python
# Hidden tests method predict in the RecSys class 

yp = cb.predict()
rmse = cb.rmse(yp)
print(rmse)
```

    1.012502820366462



```python
# tests method predict in the RecSys class 
```

# Q3. Collaborative Filtering

### 3a. Complete the function cossim in the class Collaborative [10 pts]
**To Do:**    
1.Impute the unrated entries in self.Mr to the user's average rating then subtract by the user mean, call this matrix X.   
2.Calculate cosine similarity for all item-item pairs. Don't forget to rescale the cosine similarity to be 0~1.    
You might encounter divide by zero warning (numpy will fill nan value for that entry). In that case, you can fill those with appropriate values.    

Hint: Let's say a movie item has not been rated by anyone. When you calculate similarity of this vector to anoter, you will get $\vec{0}$=[0,0,0,....,0]. When you normalize this vector, you'll get divide by zero warning and it will make nan value in self.sim matrix. Theoretically what should the similarity value for $\vec{x}_i \cdot \vec{x}_i$ when $\vec{x}_i = \vec{0}$? What about $\vec{x}_i \cdot \vec{x}_j$ when $\vec{x}_i = \vec{0}$ and $\vec{x}_j$ is an any vector?     

Hint: You may use `scipy.spatial.distance.cosine`, but it will be slow because its cosine function does vector-vector operation whereas you can implement matrix-matrix operation using numpy to calculate all cosines all at once (it can be 100 times faster than vector-vector operation in our data). Also pay attention to the definition. The scipy.spatial.distance provides distance, not similarity. 

3. Run the below cell that calculate yp and RMSE. 


```python
# Sample tests cossim method in the Collaborative class

sample_cf = Collaborative(sample_data)
sample_cf.calc_item_item_similarity(sample_cf.cossim)
sample_yp = sample_cf.predict()
sample_rmse = sample_cf.rmse(sample_yp)

assert(np.trace(sample_cf.sim)==3152), "Check cossim method in the Collaborative class. What should np.trace(cf.sim) equal?"
assert(sample_rmse==approx(1.1429596846619763, abs=5e-3)), "Check cossim method in the Collaborative class. rmse result is not as expected."
assert(sample_cf.sim[0,:3]==approx([1., 0.5, 0.5],abs=1e-2)), "Check cossim method in the Collaborative class. cf.sim isn't giving the expected results."
```


```python
# Hidden tests cossim method in the Collaborative class

cf = Collaborative(data)
cf.calc_item_item_similarity(cf.cossim)
yp = cf.predict()
rmse = cf.rmse(yp)
print(rmse)
```

    1.0263081874204125



```python
# tests cossim method in the Collaborative class 
```


```python
# additional tests for cossim method in the Collaborative class
```


```python
# additional tests for cossim method in the Collaborative class
```


```python
# additional tests for cossim method in the Collaborative class
```


```python
# additional tests for cossim method in the Collaborative class
```


```python
# additional tests for cossim method in the Collaborative class
```

### 3b. Complete the function jacsim in the class Collaborative [15 pts]
**3b [15 pts] = 3b-i) [5 pts]+3b-ii) [5 pts]+ 3b-iii) [5 pts]**

Function `jacsim` calculates jaccard similarity between items using collaborative filtering method. When we have a rating matrix `self.Mr`, the entries of Mr matrix are 0 to 5 (0: unrated, 1-5: rating). We are interested to see which threshold method works better when we use jaccard dimilarity in the collaborative filtering.    
We may treat any rating 3 or above to be 1 and the negatively rated (below 3) and no-rating as 0. Or, we may treat movies with any ratings to be 1 and ones that has no rating as 0. In this question, we will complete a function jacsim that takes a transformed rating matrix X and calculate and returns a jaccard similarity matrix.     
Let's consider these input cases for the utility matrix $M_r$ with ratings 1-5 and 0s for no-rating.    
1. $M_r \geq 3$ 
2. $M_r \geq 0$ 
3. $M_r$, no transform.

Things to think about: 
- The cases 1 and 2 are straightforward to calculate Jaccard, but what does Jaccard mean for multicategory data?
- Time complexity: The matrix $M_r$ is much bigger than the item feature matrix $M_m$, therefore it will take very long time if we calculate on dense matrix.     
Hint: Use sparse matrix.
- Which method will give the best performance?

### 3b-i)  When $M_r\geq3$ [5 pts]
After you've implemented the jacsim function, run the code below. If implemented correctly, you'll have RMSE below 0.99. 


```python
cf = Collaborative(data)
Xr = cf.Mr>=3
t0=time.perf_counter()
cf.calc_item_item_similarity(cf.jacsim,Xr)
t1=time.perf_counter()
time_sim = t1-t0
print('similarity calculation time',time_sim)
yp = cf.predict()
rmse = cf.rmse(yp)
print(rmse)
assert(rmse<0.99)
```

    similarity calculation time 0.8183888581115752
    0.9743391412966496



```python
# tests RMSE for jacsim implementation
```


```python
# additional tests for RMSE for jacsim implementation
```


```python
# additional tests for jacsim implementation
```


```python
# additional tests for jacsim implementation
```

### 3b-ii)  When $M_r\geq1$ [5 pts]
After you've implemented the jacsim function, run the code below. If implemented correctly, you'll have RMSE below 1.0. 


```python
cf = Collaborative(data)
Xr = cf.Mr>=1
t0=time.perf_counter()
cf.calc_item_item_similarity(cf.jacsim,Xr)
t1=time.perf_counter()
time_sim = t1-t0
print('similarity calculation time',time_sim)
yp = cf.predict()
rmse = cf.rmse(yp)
print(rmse)
assert(rmse<1.0)
```

    similarity calculation time 1.0006140570621938
    0.9906605122247982



```python
# tests RMSE for jacsim implementation 
```


```python
# tests RMSE for jacsim implementation
```


```python
# tests jacsim implementation
```


```python
# tests performance of jacsim implementation
```

### 3b-iii)  When $M_r$; no transform [5 pts]
After you've implemented the jacsim function, run the code below. If implemented correctly, you'll have RMSE below 0.96


```python
cf = Collaborative(data)
Xr = cf.Mr.astype(int)
t0=time.perf_counter()
cf.calc_item_item_similarity(cf.jacsim,Xr)
t1=time.perf_counter()
time_sim = t1-t0
print('similarity calculation time',time_sim)
yp = cf.predict()
rmse = cf.rmse(yp)
print(rmse)
assert(rmse<0.96)
```

### 3.C Discussion [Peer Review]
Answer the questions below in this week's Peer Review assignment. <br>
1. Summarize the methods and performances: Below is a template/example.

|Method|RMSE|
|:----|:--------:|
|Baseline, $Y_p$=3| |
|Baseline, $Y_p=\mu_u$| |
|Content based, item-item| |
|Collaborative, cosine| |
|Collaborative, jaccard, $M_r\geq 3$|  |
|Collaborative, jaccard, $M_r\geq 1$|  |
|Collaborative, jaccard, $M_r$|  |

2. Discuss which method(s) work better than others and why.

