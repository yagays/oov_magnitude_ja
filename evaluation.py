from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

from oov import MagnitudeOOV


model = KeyedVectors.load_word2vec_format("data/jawiki.word_vectors.200d.bin", binary=True)

moov = MagnitudeOOV(word2vec=model)

query_orig = "前前前世"
query_1 = "前前前前世"  # out-of-vocabulary
query_2 = "前前前前前世"  # out-of-vocabulary

v1 = moov.query(query_1)
v2 = moov.query(query_2)

print(model.most_similar(query_orig))

for i, (q1, q2) in enumerate(zip(model.similar_by_vector(v1), model.similar_by_vector(v2))):
    print(f"{i}: {q1[0]} {q2[0]}")

cos_sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1)).item()
print(f"cosine similarity between {query_1} and {query_2}: {cos_sim}")
