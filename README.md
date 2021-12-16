# Data Science Seminar Project
#### Winter Semester 2021/22
## Simon Müller

# Topic: (Multi-)Media Recommender System

Recommending books using a Recommender System based on [goodreads.com](https://www.goodreads.com/) data from the UCSD Book Graph[^fn1]

Steps needed:
- Data Preparation: Downloading Data, reading into Pandas DataFrames
- Data Cleaning and Preprocessing: Keeping only needed data, string cleanup, etc.
- Data Model Creation: creating Word Embeddings from item data like title, description, authors, etc.
- Recommender System Model Creation: selecting a Neural-Network-based recommender system, e.g. NCF, WDN, DCN, etc.
- RS Training
- RS Evaluation

Optional:
- combining the Book data with another dataset from a different domain, e.g. MovieLens 25M[^fn2] to create a multi-media recommender system, suggesting books based on movie preferences and vice versa.
- this would need more text processing to find similarities between the datasets, e.g. by extracting tags from movies and matching them to book descriptions and goodreads shelf-names


[^fn1]: Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home.
[^fn2]: F. Maxwell Harper and Joseph A. Konstan, https://grouplens.org/datasets/movielens/25m/
