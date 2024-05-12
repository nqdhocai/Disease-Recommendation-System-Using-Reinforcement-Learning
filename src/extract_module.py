from transformers import pipeline
from qdrant_module import QDRANT

TASK = 'token-classification'
MODEL = 'Clinical-AI-Apollo/Medical-NER'
AGGREGATION_STRATEGY = "SIMPLE"

pipe = pipeline(TASK, model=MODEL, aggregation_strategy=AGGREGATION_STRATEGY)
# output type
# ['entity_group', 'score', 'index', 'word', 'start', 'end']

user_info = {
        'AGE': [],
        'FAMILY_HISTORY':[],
        'HEIGHT':[],
        'FAMILY':[],
        'WEIGHT':[],
        'SEX':[]
    }

def extract_info(user_text: str, user_info: dict = user_info, radius: int = 1, use_radius_range: bool = True, threshold: int = 0.2, qdrant_client=QDRANT) -> tuple:
    """
    Extracts information from user text.

    Args:
        user_text (str): The input text.
        user_info (dict): The user information dictionary.
        radius (int, optional): The radius for symptom extraction. Defaults to 1.
        use_radius_range (bool, optional): Whether to use radius range. Defaults to True.

    Returns:
        tuple: A tuple containing the updated user information and extracted symptoms.
    """
    key_group = "SIGN_SYMPTOM"
    related_groups = ["DETAILED_DESCRIPTION", "DISEASE_DISORDER", "BIOLOGICAL_STRUCTURE"]

    def get_token_info(text: str) -> list:
        return pipe(text)

    def extract_symptoms(token_info: list, index: int, radius: int) -> list:
        symptoms = []
        token_start = token_info[index]["start"]
        token_end = token_info[index]["end"]

        start_index = index - 1
        while start_index >= 0 and token_info[start_index]["entity_group"] not in related_groups:
            start_index -= 1
        start_index += 1
        start_first_cand = token_info[start_index]["start"]

        end_index = index
        while end_index < len(token_info) and token_info[end_index]["entity_group"] not in related_groups:
            end_index += 1
        end_index -= 1
        end_second_cand = token_info[end_index]["end"]

        symptoms.extend([user_text[start_first_cand + 1:token_end], user_text[token_start + 1:end_second_cand]])

        if not use_radius_range:
            start_index = index - radius if index - radius >= 0 else index
            end_index = index + radius + 1 if index + radius < len(token_info) - 1 else index + 1
            symptom_tokens = []
            for pos in range(start_index, end_index):
                if token_info[pos]["entity_group"] not in related_groups and pos!= index:
                    break
                symptom_tokens.append(token_info[pos]["word"])
            symptom = " ".join(symptom_tokens)
            symptoms.append(symptom)
        else:
            for area_size in range(1, radius + 1):
                start_index = index - area_size if index - area_size >= 0 else index
                end_index = index + area_size + 1 if index + area_size < len(token_info) - 1 else index + 1
                pos = start_index
                while pos < end_index:
                    if token_info[pos]["entity_group"] not in related_groups and pos!= index:
                        break
                    pos += 1
                if pos!= end_index:
                    continue
                symptom_start = token_info[start_index]["start"]
                symptom_end = token_info[end_index]["end"]
                symptoms.append(user_text[symptom_start:symptom_end + 1])

        return symptoms
    def check_true_token(tokens: list, threshold: int=threshold) -> list:
        true_token = []
        for token in tokens:
            entity_prom = qdrant_client.search_entity(token)
            entity_prom = sorted(entity_prom, key=lambda x: x.score, reverse=True)
            if entity_prom[0].score - entity_prom[1].score > entity_prom[0].score*threshold and int(entity_prom[0].id) in kg.disease:
                true_token.append(entity_prom[0].payload['entity'])
        return true_token

    token_info = get_token_info(user_text)
    user_info_labels = user_info.keys()
    symptoms = []

    for index, token in enumerate(token_info):
        token_type = token["entity_group"]
        if token_type in user_info_labels:
            user_info[token_type].append(token["word"])
            continue

        if token_type == key_group:
            symptoms.append(token["word"])
            symptoms.extend(extract_symptoms(token_info, index, radius))

    symptoms = list(set(symptoms))
    symptoms = check_true_token(symptoms)

    return user_info, symptoms




