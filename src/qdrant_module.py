from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from build_kg import KnowledgeGraph
import torch

MODEL = "fine-tuned/medical-20-0-16-jinaai_jina-embeddings-v2-small-en-100-gpt-3.5-turbo-0_9062874564"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)

CLIENT_URL = "YOUR_CLIENT_URL"
API_KEY = "YOUR_API_KEY"

class QDRANT:
    def __init__(self, CLIENT_URL=CLIENT_URL, API_KEY=API_KEY):
        self.QDRANT_CLIENT_URL = CLIENT_URL
        self.QDRANT_API_KEY = API_KEY
        self.client = QdrantClient(
            url = CLIENT_URL,
            api_key = API_KEY,
        )
        self.kg = KnowledgeGraph()

    def word2vec(self, text):
        token = tokenizer(text, return_tensors='pt')
        return model(**token).pooler_output #512

    def set_up_db(self):
        id2entity = self.kg.id2entity

        self.client.create_collection(
            collection_name="entity",
            vectors_config=VectorParams(size=512, distance=Distance.DOT),
        )
        points = []
        for id, entity in id2entity:
            entity_type = kg.id2type[id]
            point_struct = PointStruct(
                id=int(id),
                vector=self.word2vec(entity).tolist()[0],
                payload={"entity": entity, "type": entity_type}
            )
            points.append(point_struct)
        operation_info = self.client.upsert(
            collection_name = "entity",
            wait = True,
            points = points
        )
        print(operation_info)

    def search_entity(self, text, limit=5):
        query_vec = self.word2vec(text).tolist()[0]
        search_result = self.client.search(
            collection_name="entity", query_vector=query_vec, limit=limit
        )
