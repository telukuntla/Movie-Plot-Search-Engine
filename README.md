# Movie-Plot-Search-Engine

Wikipedia Movie Plots


Team Members
Shanmukha Praveen Madasu (smadasu1@uncc.edu)
Sai Krishna Telukuntla (stelukun@uncc.edu)
Divya Patel (dpate146@uncc.edu)
Overview
The goal of our project is to analyze and extract meaningful information from Wikipedia movie plots dataset.
Our information retrieval algorithm returns a similar movie titles based on an input plot description.
And also, it will predict movie genre based on the plot description.
Dataset
https://www.kaggle.com/jrobischon/wikipedia-movie-plots

We got the dataset from Kaggle. These contain plot summary description scraped from Wikipedia. The dataset contains descriptions of 34,886 movies from around the world. Column descriptions are listed below.

Release Year - Year in which the movie was released
Title - Movie title
Origin/Ethnicity - Origin of movie (i.e. American, Bollywood, Tamil, etc.)
Director - Director(s)
Plot - Main actor and actresses
Genre - Movie Genre(s)
Wiki Page - URL of the Wikipedia page from which the plot description was scraped
Plot - Long form description of movie plot
Data Preprocessing
There are so many Null and Unknown values present in the dataset.



Total Unknown values



Our dataset was too huge with more than 30K movies data after removing null value rows. So, data preprocessing was very important.
First step was to remove all the data which has more than one genre because it will just confuse our model. There are 2191 unique genres.
Then we also removed all unimportant columns like director, cast etc. Because we trained our model by finding plot’s tf-idf vector.
At last, we found top 5 genre and remove other genre movies. Also, we gave different integer label to our dataset. Top genre will get 0 and it will be followed for all genre.
Tasks Involved and Steps Implemented
Configuring Spark
Understanding problem statement
Understanding the algorithm
Fetching the data
Data Preprocessing
Implementing Index, QueryIndex, TF-IDF, Cosine Similarity on local machine
Used machine learning techniques like Multinomial Logistic Regression, Naive Bayes on local machine from the spark ML libraries.
Deploying code and data to AWS web services like S3 and EMR
Generating output
Project Report
Motivation
Everyone loves watching movies. There are some movies we like and some we don’t. Sometimes we don’t remember the name of the movies, but we know the plot. In such cases, our information retrieval algorithm would give us movie names if we describe the plot to it.

Moreover, we all love watching superhero movies. Movies within a genre share common parameters. For example, consider these two movies: Avengers Endgame and Spider-Man Far from Home. These two movies are similar, where superhero protects people from evil. We can say this by intuition. But we will use cloud computing techniques to figure out similarities between movies.

Algorithms
TF-IDF
The tf–idf is the product of two statistics, term frequency and inverse document frequency. There are various ways for determining the exact values of both statistics.
Term frequency tf(t,d), the simplest choice is to use the raw count of a term in a document, i.e., the number of times that term t occurs in document d.
The inverse document frequency is a measure of how much information the word provides, i.e., if it's common or rare across all documents. It is the logarithmically scaled inverse fraction of the documents that contain the word (obtained by dividing the total number of documents by the number of documents containing the term, and then taking the logarithm of that quotient)


Cosine Similarity
Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them.
The cosine of 0° is 1, and it is less than 1 for any angle in the interval (0,π] radians. It is thus a judgment of orientation and not magnitude: two vectors with the same orientation have a cosine similarity of 1, two vectors oriented at 90° relative to each other have a similarity of 0, and two vectors diametrically opposed have a similarity of -1, independent of their magnitude.


Machine Learning
After data Preprocessing, we stuck with just two columns plot and genre. So, now our aim was to convert all movies plots into vector and then train the model.
As you can see in the figure first few steps were for feature extraction. In which we did punctuation removal which will remove all comma, semi-colon, question-mark etc. It will be followed by removing stop words like is, are, was etc.
Then we did tokenization and vectorization to convert plot into vector of words and which helped us to calculate tf-idf vector. The output was very sparse with 70K features.
As we want to solve multiclass classification problem, we tried to train our data using Naïve Bayes and Logistic Regression. Which are both likelihood-probabilistic model which can give class probabilities about that genre to be a particular class and we can pick the best.


Libraries used
[x] Tokenizer, CountVectorizer, StopWordsRemover
[x] VectorUDT
[x] SparseVector, DenseVector
[x] NaiveBayes, Logistic Regression
[x] MulticlassClassificationEvaluator
[x] ChiSqSelector
External Tools
Amazon S3 for data storage
AWS EMR for data processing and analysis
Framework
Apache Spark
Implemented TF-IDF and Cosine similarity algorithms using Spark
Used Logistic Regression and Naive Bayes from spark ml libraries


Hadoop
Implemented Index and QueryIndex algorithms using hadoop



Expectations/Aspects
1. What to expect?
Implementation of TF-IDF & Cosine similarity algorithms to extract movie similarities from plot summaries.
We did data preprocessing and feature extraction on huge dataset
We solved multiclass classification problem and made system which can predict genre from the plot.
We made whole system in a distributive way using pyspark.
We used two different machine learning algorithms like Naïve Bayes and logistic regression from spark ml libraries. Also, made comparison with each other.
2. Likely to accomplish
Predicting movie genre based on plot description using machine learning algorithms with high accuracy.
3. Ideal Accomplishments
Suggested modifications in the exisitng implementation.
Extract the data using web crawling technique.
Team Responsibilities
Task	Team Members
Preprocessing the Data	Praveen Madasu, Divya Patel
Implementation of TF-IDF & Cosine Similarity algorithms	Sai Krishna Telukuntla, Praveen Madasu
Machine learning algorithms using spark ml libraries	Divya Patel, Praveen Madasu
Project Report	Sai Krishna Telukuntla, Divya Patel, Praveen Madasu
Results
Query


Top 5 Similar movie names


Logistic Regerssion Accuracy


Naive Bayes Accuracy


Model	For 3 class	For 5 class	For 10 class
Naive Bayes	62%	53%	51%
Logistic Regression	54%	52%	54%
As you can see, we have got 62% accuracy using multinomial naïve bayes and 54% at max which is not that great because as we said in the challenge due to lot of unique genres and most of them are kind of overlapping on each other.
It is more of multilabel classification problem than multiclass problem.
Confusion Matrix
Drama	Comedy	Horror
Drama	1276	680	358
Comedy	407	1064	179
Horror	60	52	321
From this, we can say that many times (1087) model confused between comedy and drama. Not confusing with horror. Also, many horror movies are predicted as drama as most of them can be called drama horror. As we have more plot of comedy and drama and their plots are more general, it's kind of dominanting. You can say our model is little bit biased about them.

Challenges Faced
Unfortunately, there are lots of challenges we faced during our project.
Firstly, dataset was too big with very huge plots of every movie. So, vocabulary size was around 70K. It made our output vector very huge and sparse. Which made hard during training. We got timed-out error many times then instead of parsing that sparse matrix, we converted matrix in dense manner. Which has tuple with three values. First one is total feature, second is list of indexes of non-zero tf-idf score words and last value of score.
We have 2K unique genres with lots of genre combining other two genre which confused our model while learning. Like lots of drama movies are comedy, too. This makes this huge dataset kind of small in manner of induvial genre wise.
Conclusion
Using TF-IDF and Cosine similarity algorithms, we have succesfully extracted similar movies list with the user plot description.
And have successfully calssified movie genres based on movie plot description using logistic regression and naive bayes from spark ml libraries.
References¶
TF-IDF
Cosine Similarity
spark documentation
Classification documentation
