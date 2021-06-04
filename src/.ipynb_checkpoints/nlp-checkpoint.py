import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import re
import numpy as np
import tensorflow_recommenders as tfrs
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
data = pd.read_csv("../ml-25m/movies.csv")

# define some useful helper functions
# these will remove the parenthesis from the titles
# and the pipes between genres for each item and finally
# remove the null items for movies with no genre
def remove_pars(x):
    x = str(x)
    return re.sub('[()]', "", x)

def remove_pipes(x):
    x = str(x)
    return re.sub('\|', " ", x)

def remove_nulls(a, b, i):
    string_m = a[i] + " " + b[i]
    return re.sub("\(no genres listed\)", "", string_m)

# process the titles and genres
titles = [remove_pars(i) for i in data['title']]
genres = [remove_pipes(i) for i in data['genres']]

# make a list of the input strings for each item from these bits of data. 
input_string = [remove_nulls(titles, genres, i) for i in range(len(genres))]

# this will be using a GPU to speed things up 
# but will default to CPU if no devices are available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Create embeddings for each item 
embeddings_list = []
for _, i in enumerate(input_string):
    encoded_input = tokenizer(i, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.pooler_output
    embeddings = torch.nn.functional.normalize(embeddings)
    embeddings_list.append(embeddings)
    if _ % 10000  == 0:
        print(str(_))
        
# extract the embeddings
embeddings_list_tensors = []
for i in embeddings_list:
    d = i.cpu()[0].numpy()
    embeddings_list_tensors.append(d)

# save them to local file. 
embeddings = pd.DataFrame(np.vstack(embeddings_list_tensors))
embeddings.to_csv("../embeddings/data.csv")