import torch
from train_agent import DQN
from environment import Env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Khởi tạo môi trường
env = Env()
input_size = env.state_embed_size

# Khởi tạo mô hình
model = DQN(input_size)  # Truyền input_size vào hàm khởi tạo của model
model.load_state_dict(torch.load("dqn-policy.pt"))
model.eval()
print(model)


