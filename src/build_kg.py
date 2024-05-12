import json
import pickle
import os
import random


class KnowledgeGraph:
    def __init__(self, kg_dir=""):
        self.kg_dir = kg_dir
        self.kg_info_path = os.path.join(kg_dir, 'kg_info_en.pkl')
        self.set_kg_info()
    def set_kg_info(self):
        RELATED_SYMPTOM_ID = 0
        CAUSED_ID = 1
        RELATED_DISEASE_ID = 14

        print('Loading knowledge graph...')
        with open(self.kg_info_path, 'rb') as f:
            kg_info = pickle.load(f)

        print('Loading Graph...')
        setattr(self, 'graph', kg_info['graph'])
        print('Loading Sub Graph...')
        setattr(self, 'sub_graph', kg_info['sub_graph'])
        print('Loading Entities...')
        setattr(self, 'entity2id', kg_info['entity2id'])
        setattr(self, 'id2entity', kg_info['id2entity'])
        print('Loading Relations...')
        setattr(self, 'relation2id', kg_info['relation2id'])
        setattr(self, 'id2relation', kg_info['id2relation'])
        print('Loading Graph Adjacency...')
        setattr(self, 'graph_adj', kg_info['graph_adj'])

        print('Updating Relations...')
        self.set_symptom_disease_relation()

        disease_ids = []
        symptom_ids = []
        cause_ids = []
        medicine_ids = []

        for entity_id in self.graph.keys():
            e_relation_id = self.graph[entity_id].keys()
            if RELATED_DISEASE_ID in e_relation_id:
                disease_ids.append(entity_id)
                disease_ids.extend(self.graph[entity_id][RELATED_DISEASE_ID])

            if RELATED_SYMPTOM_ID in e_relation_id:
                if entity_id not in disease_ids:
                    disease_ids.append(entity_id)
                symptom_ids.extend(self.graph[entity_id][RELATED_SYMPTOM_ID])

            if CAUSED_ID in e_relation_id:
                if entity_id not in disease_ids:
                    disease_ids.append(entity_id)
                cause_ids.extend(self.graph[entity_id][CAUSED_ID])
        disease_ids = list(set(disease_ids))
        disease_ids.sort()
        symptom_ids = list(set(symptom_ids))
        symptom_ids.sort()
        cause_ids = list(set(cause_ids))
        cause_ids.sort()
        setattr(self, 'disease', disease_ids)
        setattr(self, 'symptom', symptom_ids)
        setattr(self, 'cause', cause_ids)

        print('Loading Done !!')
        print(f'KG has {len(self.graph)} entities | {len(self.id2relation)} relations')

    def set_symptom_disease_relation(self):
        RELATED_SYMPTOM_ID = 0
        SYMPTOM_TO_DISEASE_ID = int(max(self.id2relation.keys())) + 1
        self.id2relation[SYMPTOM_TO_DISEASE_ID] = 'symptom to disease'
        self.relation2id['symptom to disease'] = SYMPTOM_TO_DISEASE_ID

        related_symptom_graph = {}
        for entity_id in self.graph:
            e_relation_id = self.graph[entity_id].keys()
            if RELATED_SYMPTOM_ID in e_relation_id:
                related_symptom_graph[entity_id] = self.graph[entity_id][RELATED_SYMPTOM_ID]

        symptom = []
        for disease in related_symptom_graph.keys():
            symptom.extend(related_symptom_graph[disease])
        symptom = list(set(symptom))
        symptom_to_disease_graph = {i:[] for i in symptom}

        for disease in related_symptom_graph.keys():
            for s in related_symptom_graph[disease]:
                symptom_to_disease_graph[s].append(disease)

        for symptom in symptom_to_disease_graph.keys():
            if symptom_to_disease_graph[symptom] != []:
                self.graph[symptom][SYMPTOM_TO_DISEASE_ID] = symptom_to_disease_graph[symptom]


kg = KnowledgeGraph()
print(kg.id2entity[3995])


