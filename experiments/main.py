
import torch
import pickle
import random
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence

def get_data(filename):
  pickle_in = open(filename,"rb")
  return pickle.load(pickle_in)

def get_embedder():
  glove_embedding = WordEmbeddings('glove')
  flair_embedding_forward = FlairEmbeddings('news-forward')
  flair_embedding_backward = FlairEmbeddings('news-backward')
  
  return DocumentPoolEmbeddings([glove_embedding, flair_embedding_backward, flair_embedding_forward], pooling='mean')
  
def get_embedding(content, embedder=get_embedder()):
  paragraph = Sentence(content)
  embedder.embed(paragraph)
  return paragraph.get_embedding().unsqueeze(0)

def generate_embeddings_tensor(data, limit=5, save_filename='model.pt'):

  keys = sorted(list(d.keys()))[:5]
  result = torch.zeros(1, 4196)
  embedder = get_embedder()

  for i in range(len(keys)):
    print(keys[i])
    content = data[keys[i]]
    embedding = get_embedding(content, embedder)
    result = torch.cat([result, embedding], dim=0)

  torch.save(result.narrow(0, 1, len(keys)), save_filename)


def find_similar_content_byID(id,data_filename='data/data.pkl'):
  d = get_data(data_filename)
  return find_similar_content(d[id], d)


def find_similar_content(content, data, topk=5, model_filename='model.pt'):

  a = get_embedding(content)
  b = torch.load(model_filename)

  #Compute cosine similatiry: smaller the angle = higher cosine similarity
  a_norm = a / a.norm(dim=1)[:, None]
  b_norm = b / b.norm(dim=1)[:, None]
  res = torch.mm(a_norm, b_norm.transpose(0,1))

  #Get Top k similar items
  scores, indexes = torch.topk(res, topk, dim=1)
  scores, indexes = scores.tolist()[0], indexes.tolist()[0]
  
  #Print Results
  for i in range(topk):
    print(f"{i+1}: score {scores[i]}, content_id {indexes[i]}")
    print(data[indexes[i]])
    print("\n\n\n")




d = get_data("data/data.pkl")

#Generate the embeddings for all the paragraphs
generate_embeddings_tensor(d)

#Find similar content

#find_similar_content_byID(1)

#OR

#find_similar_content("This is a random query", d)



