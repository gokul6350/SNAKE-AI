import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)  # Fixed the linear layer here

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x



    def load(self, file_path, optimizer=None):
        checkpoint = torch.load(file_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save(self, file_name='Model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class Qtriner:
    def __init__(self,model,lr,gamma):
        self.lr=lr
        self.gamma=gamma
        self.model=model
        self.optimizer=optim.Adam(model.parameters(),lr=self.lr)
        self.criton=nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
    
        #if action == None:
         #   action = 0
        action = torch.tensor(action, dtype=torch.long)
  

        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criton(target, pred)  # Fixed the typo here (self.criton -> self.criterion)
        loss.backward()
        self.optimizer.step()
"""model = Linear_QNet(11,256,3)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the model and optimizer's state from the checkpoint
model.load(file_path='./model/Model.pth', optimizer=optimizer)
"""
