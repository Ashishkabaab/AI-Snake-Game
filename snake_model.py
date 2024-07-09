import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QModel(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        #three values returned in list
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        #cost function
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, gameover):
        
        #convert features to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:

            # if we have only one dimension
            # we want to append one dimension
            # we want the form (1, x) (would be len=2)
            #batch would be of the form (n, x)

            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            #defining a tuple with one value
            gameover = (gameover, )
        
        #Bellman Equation

        # Want to get predicted Q values with current state
        #Three different values representing different actions
        pred = self.model(state)

        target = pred.clone()
        for index in range(len(gameover)):
            Q_new = reward[index]
            if not gameover[index]:
                #if game not over, new Q value = currentReward + gamma*(maximum value (based on action) of next state)
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))

            #setting maximum value of action (at current index) to new Q value
            target[index][torch.argmax(action[index]).item()] = Q_new

        # pred.clone()
        # target[argmax(action)] = Q_new

        #reset gradient
        self.optimizer.zero_grad()
        #loss function: MSE -> (Qnew - Q)^2
        loss = self.criterion(target, pred)
        #Backpropogation
        loss.backward()
        #Gradient Descent
        self.optimizer.step()


