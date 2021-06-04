import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import tensorflow_recommenders as tfrs
import tensorflow as tf
from tensorflow import keras

class personalisedSearcher:
    def __init__(self):
        self.movies = pd.read_csv("ml-25m/movies.csv")
        self.ratings = pd.read_csv("ml-25m/ratings.csv")
        self.embeddings = pd.read_csv("embeddings/data.csv", index_col=0)
        self.item_tensor = tf.convert_to_tensor(self.embeddings, dtype=tf.float32)
        self.scann = tfrs.layers.factorized_top_k.ScaNN(num_leaves=1000, 
                                                        num_leaves_to_search = 100, 
                                                        k = round(np.sqrt(len(self.item_tensor))))
        self.scann.index(self.item_tensor)
        self.model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
        self.recommender = keras.models.load_model('CF')
        
    def get_user_encodings(self):
        user_ids = self.ratings["userId"].unique().tolist()
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        userencoded2user = {i: x for i, x in enumerate(user_ids)}
        
        return user2user_encoded, userencoded2user

    def get_movie_encodings(self):
        movie_ids = self.ratings["movieId"].unique().tolist()
        movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
        movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
        
        return movie2movie_encoded, movie_encoded2movie
    
    def update_ratings(self):
        user2user_encoded, _ = self.get_user_encodings()
        movie2movie_encoded, _ = self.get_movie_encodings()
        self.ratings["user"] = self.ratings["userId"].map(user2user_encoded)
        self.ratings["movie"] = self.ratings["movieId"].map(movie2movie_encoded)
        
        return self.ratings
        
    def get_user_history(self, user_id):
        df = self.update_ratings()
        watched_movies = df[df.userId == user_id]
        
    def get_candidate_movies(self, query):
        encoded_input = self.tokenizer(query, 
                                  padding=True, 
                                  truncation=True, 
                                  max_length=64, 
                                  return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        query_embeddings = model_output.pooler_output
        query_embeddings = torch.nn.functional.normalize(query_embeddings)
        test_case = self.scann(np.array(query_embeddings))
        return self.movies.iloc[test_case[1].numpy()[0]][0:11]
    
    def filter_candidates(self, user_id, query):
        movies_watched_by_user = self.ratings[self.ratings.userId == user_id]
        candidates = self.get_candidate_movies(query)
        movies_not_watched = candidates[
            ~candidates["movieId"].isin(movies_watched_by_user.movieId.values)
        ]["movieId"]
        movie2movie_encoded, _ = self.get_movie_encodings()
        movies_not_watched = list(set(movies_not_watched).
                                  intersection(set(movie2movie_encoded.keys())))
        movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
        user2user_encoded, _ = self.get_user_encodings()
        user_encoder = user2user_encoded.get(user_id)
        movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), movies_not_watched))
        
        return movie_array, movies_not_watched, movies_watched_by_user
    
    def personalised_search(self, user_id, query):
        movie_array, movies_not_watched, movies_watched_by_user = self.filter_candidates(user_id, query)
        scored_items = self.recommender.predict(movie_array).flatten()
        top_rated = scored_items.argsort()[-10:][::-1]
        _, movie_encoded2movie = self.get_movie_encodings()
        recommended_movie_ids = [movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_rated]
        
        return recommended_movie_ids, movies_watched_by_user
    
    def print_recs(self, user_id, query):
        recommendations, movies_watched_by_user = self.personalised_search(user_id, query)
        
        print("Showing recommendations for user: {}".format(user_id))
        print("====" * 9)
        print("Movies with high ratings from user")
        print("----" * 8)
        top_movies_user = (
            movies_watched_by_user.sort_values(by="rating", ascending=False)
            .head(5)
            .movieId.values
        )
        movie_df_rows = self.movies[self.movies["movieId"].isin(top_movies_user)]
        for row in movie_df_rows.itertuples():
            print(row.title, ":", row.genres)
        print("----" * 8)
        print("Top movie recommendations")
        print("----" * 8)
        recommended_movies = self.movies[self.movies["movieId"].isin(recommendations)]
        for row in recommended_movies.itertuples():
            print(row.title, ":", row.genres)