import torch
import random
from build_kg import KnowledgeGraph
from dataset import Dataset

reward_dict = {
            "ask_suc": 0.01,
            "ask_fail": -0.01,
            "rcm_suc_all": 1,
            "rcm_fail": -1,
            "until_T": -0.3,
            "rcm_suc_any": 0.3
        }
class Env:
    def __init__(self, kg=KnowledgeGraph(), dataset=Dataset(), max_turn=15, score_threshold=0.1, mode='train', reward_dict=reward_dict):
        self.max_turn = max_turn
        self.kg = kg
        self.score_threshold = score_threshold
        self.dataset = dataset
        self.disease_length = len(self.dataset.disease)
        self.symptom_length = len(self.dataset.symptom)
        self.state_embed_size = self.disease_length + self.symptom_length #+ self.max_turn
        self.done = False
        self.mode = mode

        # user profile
        self.symptom = []
        self.related_symptom = []
        self.reachable_symptom = []
        self.cand_disease = [] # [(dis_id, dis_score)]
        self.disease_symptom_pair = {} # {dis_id: {sym_id: weight}}

        # target.
        self.start_disease = []
        self.disease_target = None
        self.pred_disease = []

        # state vector
        self.cur_conversation_step = 0
        self.conversation_his = []

        self.action_space = {
            0: "ask_more",
            1: "rec_disease"
        }

        self.reward_dict = reward_dict
    def reset(self, user_sym=[]):
        self.cur_conversation_step = 0
        self.related_symptom = []
        self.related_symptom_score = []
        self.reachable_symptom = []
        self.cand_disease = []
        self.cand_disease_score = []
        self.pred_disease = []
        self.disease_symptom_pair = {}

        if self.mode == 'train':
            start_sym_num = random.randint(1, 3)
            self.symptom = random.choices([self.dataset.subID2trueID[i] for i in self.dataset.symptom.values()], k=start_sym_num)
            related_disease = []
            KG_SYMPTOM_TO_DISEASE_ID = 29
            for sym in self.symptom:
                dis_cand = []
                for i in self.kg.graph[sym][KG_SYMPTOM_TO_DISEASE_ID]:
                    if i in self.dataset.subID2trueID.values():
                        dis_cand.append(i)
                related_disease.extend(dis_cand)
            self.disease_target = random.choices(related_disease, k=1)
        else:
            self.symptom.extend(user_sym)

        self._update_cand_disease()
        self._update_cand_symptom()
        self.cur_conversation_step += 1

        return self._get_state()

    def _update_cand_disease(self):
        cand_disease = []
        DT_SYMPTOM_TO_DISEASE_ID = 1
        for sym in self.symptom:
            sub_sym_id = self.dataset.trueID2subID[sym]
            if sub_sym_id in self.dataset.graph:
                for dis in self.dataset.graph[sub_sym_id].get(DT_SYMPTOM_TO_DISEASE_ID, []):
                    weight = abs(self.dataset.graph_adj[sub_sym_id][dis])
                    if self.dataset.subID2trueID[dis] not in self.disease_symptom_pair:
                        self.disease_symptom_pair[self.dataset.subID2trueID[dis]] = {}
                    if sym not in self.disease_symptom_pair[self.dataset.subID2trueID[dis]]:
                        self.disease_symptom_pair[self.dataset.subID2trueID[dis]][sym] = weight
        sym_num = len(self.symptom)
        for dis, sym_weight in self.disease_symptom_pair.items():
            sym_freq = len(sym_weight.keys())
            dis_score = sum(sym_weight.values()) * (sym_freq / sym_num)
            cand_disease.append((dis, dis_score))
        self.cand_disease = cand_disease

    def _update_cand_symptom(self):
        cand_sym = []
        RELATED_SYMPTOM_ID = 0
        for dis in self.cand_disease:
            cand_sym.extend(self.kg.graph[dis[0]][RELATED_SYMPTOM_ID])
        cand_sym = list(set(cand_sym))
        for sym in cand_sym:
            if sym not in self.reachable_symptom:
                self.reachable_symptom.append(sym)

    def _get_state(self):
        user_embed = [0 for _ in range(self.state_embed_size)]
        for sym in self.symptom:
            sym_sub = self.dataset.trueID2subID[sym]
            if sym_sub in list(self.dataset.symptom.values()):
                sym_index = list(self.dataset.symptom.values()).index(sym_sub)
                user_embed[sym_index] = 1
        for cand in self.cand_disease:
            cand_i = self.dataset.trueID2subID[cand[0]]
            dis_index = list(self.dataset.disease.values()).index(cand_i)
            user_embed[dis_index + self.symptom_length] = cand[1] # disease score
        # for i, hist_info in enumerate(self.conversation_his):
        #     if hist_info is None:
        #         id = 0
        #     else:
        #         id = hist_info
        #     user_embed[i + self.disease_length + self.symptom_length] = id  # conversation history info

        return torch.tensor(user_embed)

    def step(self, action, user_sym=[]):
        done = False
        if self.mode=='test':
            print(f"--------------STEP {self.cur_conversation_step}--------------")
        if self.cur_conversation_step == self.max_turn:
            done = True
            reward = self.reward_dict['until_T']
        elif action == 0:
            if self.mode == 'train':
                reward, done = self._ask_update(self._get_user_sym())
            else:
                reward, done = self._ask_update(user_sym)
        else: # recommend disease
            done = True
            rcm_disease = []
            cand_num = len(self.cand_disease)
            self.cand_disease.sort(key= lambda x: x[1], reverse=True)
            if cand_num == 1:
                rcm_disease == self.cand_disease
            else:
                highest_score_dis = self.cand_disease[0]
                score_threshold = highest_score_dis[1] * self.score_threshold
                for rank, cand in enumerate(self.cand_disease):
                    if abs(cand[1] - highest_score_dis[1]) <= score_threshold:
                        rcm_disease.extend([cand])
                    else:
                        rcm_disease.append(cand)
                        break
                rcm_disease = list(set(rcm_disease))
            self.pred_disease = rcm_disease
            reward = 0
            if self.mode == 'train':
                rcm_disease_id = [i[0] for i in rcm_disease]
                if all(i in self.disease_target for i in rcm_disease_id):
                    reward = self.reward_dict['rcm_suc_all']
                elif any(i in self.disease_target for i in rcm_disease_id):
                    reward = self.reward_dict['rcm_suc_any']
                else:
                    reward = self.reward_dict['rcm_fail']
        self.cur_conversation_step += 1
        return self._get_state(), reward, done

    def _get_user_sym(self):
        num_user_sym = [i for i in range(11)]
        weight = [i / 1000 if i <= 5 else i / 10 for i in range(1, 12)]
        weight.sort(reverse=True)
        num_user_sym = random.choices(num_user_sym, weight, k=1)

        rand_sym = [self.dataset.subID2trueID[i] for i in self.dataset.symptom.values()]
        for sym in self.reachable_symptom:
            if sym in rand_sym:
                rand_sym.remove(sym)
        for sym in self.symptom:
            if sym in rand_sym:
                rand_sym.remove(sym)
        num_sym_out = random.randint(0, len(self.reachable_symptom) // 4 + 3)
        rand_sym = random.choices(rand_sym, k=num_sym_out)
        rand_sym.extend(self.reachable_symptom)
        if num_user_sym[0] < len(rand_sym):
            rand_sym = random.choices(rand_sym, k=num_user_sym[0])
        return rand_sym

    def _ask_update(self, user_sym:list):
        done = False
        if all(i in self.symptom for i in user_sym):
            reward = self.reward_dict['ask_fail']
        else:
            reward = self.reward_dict['ask_suc']
            self.symptom.extend(user_sym)
            self.symptom = list(set(self.symptom))
            self._update_cand_symptom()
            self._update_cand_disease()
        return reward, done

    def _pred_diseases(self):
        return self.pred_disease

