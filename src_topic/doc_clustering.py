from common_module import *
import hdbscan
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN

class documentClustering:
    
  def __init__(self, data_path):
    # instantiate scaler object
    if os.path.exists(os.path.join(data_path, "models/scaler_model.pkl")):
        self.scaler = pickle.load(open(os.path.join(data_path, "models/scaler_model.pkl"), 'rb'))
    else:
        self.scaler = StandardScaler()

    self.clusterer = pickle.load(open(os.path.join(data_path, "models/clusterer_model.pkl"), 'rb'))


  def train_cluster(self, train_df):
    df = self.scaler.fit_transform(train_df)
    model = HDBSCAN(min_cluster_size=100, min_samples=1, metric='euclidean', cluster_selection_method='eom', gen_min_span_tree=True, prediction_data=True).fit(df)

    return model, self.scaler

  def test_cluster(self, test_df):
    df = self.scaler.transform(test_df)
    test_labels, strengths = hdbscan.approximate_predict(self.clusterer, df)
    return test_labels
