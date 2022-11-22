from common_module import *
from string import punctuation
from sentence_transformers import SentenceTransformer
from umap import UMAP

# Loading NLTK Modules
from nltk.tokenize import sent_tokenize

class documentEmbedding:
    
  '''

  This class can be used online (in colab) or offline (locally):
  1. Online:
  If using this class in Colab and downloading the data from external source using the code
  in the notebook only run download_dataset function above in the code cell.
  2. Offline:
  If using this class to process news article data available in a local directory,
  "data_path" parameter should be defined.
  Where "data_path" is the path to the folder containing all news articles datasets
  datasets:

  Parameters:
  -----------

  data_path: str
  the path to 'all-the-news-2-1.csv' data if the data is downloaded from GDrive or other location.

  '''

  def __init__(self, data_path):
    # self.data = df
    self.sentence_model = SentenceTransformer("all-mpnet-base-v2")
    self.umap_reducer = pickle.load(open(os.path.join(data_path, "models/umap_reducer_model.sav"), 'rb'))

  def doc_clean(self, text):
    text = text.lower()
    text = text.replace('\xa0', '')
    text = re.sub('[!"#$%&\'()’*+,-/:;<=>?—@[\\]^_`{|}~’]', '', text)
    return text

  def sentence_to_vector(self, sent):
      # Encode the sentence
      embeddings = self.sentence_model.encode(sent, show_progress_bar=False)

      return embeddings
    
  def doc_to_vectors(self, doc):
      doc = self.doc_clean(str(doc))
      sentences  = sent_tokenize(doc)
      # sentence to vector representation
      vector = [self.sentence_to_vector(sent) for sent in sentences]
      doc_embd = np.multiply.reduce(vector)

      return doc_embd

  def generate_embedding(self, data_df):
    print("Generating embedding vectors ...")
    data_df['article_embd'] = data_df['article'].apply(self.doc_to_vectors)
    df_vec = pd.DataFrame([list(emb) for emb in data_df['article_embd'].values])
    return df_vec

  def feature_reduction(self, embd_vector, umap_reducer=False):
    if umap_reducer:
      reducer = umap_reducer
      data_umap = reducer.transform(embd_vector)
    else:
      reducer =  UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
      reducer.fit(embd_vector)
      data_umap = reducer.transform(embd_vector)
    return pd.DataFrame(data_umap), reducer 

  def embedding_main(self, df):
    df_vec = self.generate_embedding(df)
    df_vec, _ = self.feature_reduction(df_vec,umap_reducer=self.umap_reducer)
    return df_vec
        
        
        