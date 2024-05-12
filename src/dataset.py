import pickle
class Dataset:
    def __init__(self, path='data\dataset.pkl'):
        self.path = path
        self.data = self._load_data()

    def _load_data(self):
        with open(self.path, 'rb') as f:
            dt = pickle.load(f)

        setattr(self, 'graph', dt['graph'])
        setattr(self, 'disease', dt['disease'])
        setattr(self, 'symptom', dt['symptom'])
        setattr(self, 'subID2trueID', dt['subID2trueID'])
        setattr(self, 'trueID2subID', dt['trueID2subID'])
        setattr(self, 'graph_adj', dt['graph_adj'])

        return dt