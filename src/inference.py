import torch
from train_agent import DQN
from environment import Env
from extract_module import extract_info

# Khởi tạo môi trường
env = Env()
input_size = env.state_embed_size



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    MAX_TURN_REWARD = -0.3
    
    env = Env('test')
    input_size = env.state_embed_size
    
    model = DQN(input_size)
    model.load_state_dict(torch.load('dqn-policy.pt'))
    model.eval()
    
    userText = str(input())
    _, userSymptoms = extract_info(userText)
    state = env.reset(userSymptoms)
    reachableSymptom = env.reachable_symptom
    action = model.select_action(state)
    state, reward, done = env.step(action)
    
    while not done:
        userText = str(input())
        _, userSymptoms = extract_info(userText)
        
        action = model.select_action(state)
        state, reward, done = env.step(action, userSymptoms)
        if not done:
            print(f'Pls give me some symptoms, such as: {env.reachable_symptom[0]},...')
    
    if reward == MAX_TURN_REWARD:
        print('Can\'t find your Disease !')
    else:
        print(f'Predict Diseases: {env.pred_disease}')
            
    
    


