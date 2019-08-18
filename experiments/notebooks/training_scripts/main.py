
import torch
import pickle
import random
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence
import argparse
from azureml.core import Run


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_folder', default='data/', help='dataset folder')
parser.add_argument('--size', type=int, default=100, help='size to process')
opt = parser.parse_args()
print(opt)


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

def generate_embeddings_tensor(run, data, limit=5, save_filename='model.pt'):

  keys = sorted(list(d.keys()))[:limit]
  result = torch.zeros(1, 4196)
  embedder = get_embedder()

  print("Embedding Started")

  try:
    for i in range(len(keys)):
      print(keys[i])
      content = data[keys[i]]
      embedding = get_embedding(content, embedder)
      result = torch.cat([result, embedding], dim=0)

    print("Finished")
  except:
    print("Exception occurred.")
    pass
    
  filename = f'outputs_{opt.size}/{save_filename}'
  os.makedirs(f"outputs_{opt.size}", exist_ok=True)
  torch.save(result.narrow(0, 1, len(keys)),filename)

  print("Saved")

  run.upload_file(name=save_filename, path_or_stream=filename)
  run.complete()


if __name__ == '__main__':
  run = Run.get_context()
  d = get_data(os.path.join(opt.dataset_folder, 'data.pkl'))

  #Generate the embeddings for all the paragraphs
  generate_embeddings_tensor(run, d, limit=opt.size)


