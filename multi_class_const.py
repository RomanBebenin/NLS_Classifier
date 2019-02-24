from gensim.models import KeyedVectors
import pandas as pd
customer_df = pd.read_csv('Consumer_Complaints.csv')
_word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
_vector_dim = 300 # const for the dimension of vectors in Word2Vec model
