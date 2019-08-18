# 📑 Similar Content Service - Heartathon 2019

Applied Flair [Word](https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md)+[Document Embeddings](https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md) on a small subset of the given mission literature dataset. Then computed cosine similarity on the embedding vectors. Top 'k' elements from resulting vector are mapped with the content id's and sent back as 'Similar Content' in an REST API.

Tech Stack includes Python ([pytorch](https://github.com/pytorch/pytorch), [flair](https://github.com/zalandoresearch/flair), [pandas](https://github.com/pandas-dev/pandas)) + [Azure Machine Learning Service](https://azure.microsoft.com/en-gb/services/machine-learning-service/) for training in cloud and model deployment as webservice (training on full dataset is in progress).

### Complete API Documentation

https://documenter.getpostman.com/view/5756089/SVfGzBy2?version=latest

### References

1. [Flair: State-of-the-Art Natural Language Processing Library (NLP)](https://www.aclweb.org/anthology/N19-4010)
2. [Contextual String Embeddings for Sequence Labelling](https://alanakbik.github.io/papers/coling2018.pdf)
3. [Text Similarities : Estimate the degree of similarity between two texts](https://medium.com/@adriensieg/text-similarities-da019229c894)
4. [Quick review on Text Clustering and Text Similarity Approaches](http://www.lumenai.fr/blog/quick-review-on-text-clustering-and-text-similarity-approaches)


