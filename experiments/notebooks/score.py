import json
import os
import pickle
import torch
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
  paragraph = Sentence(str(content))
  embedder.embed(paragraph)
  return paragraph.get_embedding().unsqueeze(0)

def find_similar_content_byID(id):
  return find_similar_content(data[id])


def find_similar_content(content, topk=10):

  a = get_embedding(content)
  b = model

  #Compute cosine similatiry: smaller the angle = higher cosine similarity
  a_norm = a / a.norm(dim=1)[:, None]
  b_norm = b / b.norm(dim=1)[:, None]
  res = torch.mm(a_norm, b_norm.transpose(0,1))

  #Get Top k similar items
  scores, indexes = torch.topk(res, topk, dim=1)
  scores, indexes = scores.tolist()[0], indexes.tolist()[0]
  
  response = []

  for i in range(topk):
    d = {'id':indexes[i], 'cos_similarity_score':scores[i], 'content':data[indexes[i]]}
    response.append(d)
  
  return response

def init():

  global model, data
  data_filename='outputs/data.pkl'
  model_filename='outputs/model.pt'

  data = get_data(data_filename)
  model = torch.load(model_filename)

def run(raw_data):
  
  req = json.loads(raw_data)

  try:
    id = int(req['id'])
    res = find_similar_content_byID(id)
  except:
    query = req['query']
    res = find_similar_content(query)
  
  return res