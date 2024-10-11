
import random
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import io
from torch.utils.data import Dataset, TensorDataset
from pytorchtools import EarlyStopping

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed = 0
seed_everything(seed)


x = np.loadtxt('logV_nobigSTD_norm.txt')
y = np.loadtxt('logR_norm.txt')
x_length = x.shape[1]
y_length = y.shape[1]
dn = x.shape[0]
x_data = torch.tensor(x[:,:], dtype=torch.float32)
y_data = torch.tensor(y[:,:], dtype=torch.float32)
batch_size = 400
train_dataset = TensorDataset(x_data, y_data)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True)


class MDN(nn.Module):
    def __init__(self, n_hidden_in, n_gaussian_in, stride):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(int(n_hidden_in / 4), n_hidden_in),
            nn.Tanh()
        )
        self.l_x = nn.Linear(1, n_hidden_in)
        self.l_x2 = nn.Linear(38, n_hidden_in)
        self.c_x = nn.Conv1d(80, y_length, kernel_size=3, stride=stride)
        self.z_alpha = nn.Sequential(
            nn.Linear(n_hidden_in, n_hidden_in),
            nn.ReLU(),
            nn.Linear(n_hidden_in, n_hidden_in),
            nn.ReLU(),
            nn.Linear(n_hidden_in, n_hidden_in),
            nn.ReLU(),
            nn.Linear(n_hidden_in, n_hidden_in),
            nn.ReLU(),
            nn.Linear(n_hidden_in, n_gaussian_in)
        )
        self.z_mu = nn.Sequential(
            nn.Linear(n_hidden_in, n_hidden_in),
            nn.ReLU(),
            nn.Linear(n_hidden_in, n_hidden_in),
            nn.ReLU(),
            nn.Linear(n_hidden_in, n_hidden_in),
            nn.ReLU(),
            nn.Linear(n_hidden_in, n_hidden_in),
            nn.ReLU(),
            nn.Linear(n_hidden_in, n_gaussian_in)
        )
        self.z_sigma = nn.Sequential(
            nn.Linear(n_hidden_in, n_hidden_in),
            nn.ReLU(),
            nn.Linear(n_hidden_in, n_hidden_in),
            nn.ReLU(),
            nn.Linear(n_hidden_in, n_hidden_in),
            nn.ReLU(),
            nn.Linear(n_hidden_in, n_hidden_in),
            nn.ReLU(),
            nn.Linear(n_hidden_in, n_gaussian_in),
            nn.ELU()
        )
        self.lstm1 = torch.nn.LSTM(38, 50)
        self.lstm2 = torch.nn.LSTM(50, 80)
        self.lstm3 = torch.nn.LSTM(80, 50)
        self.fc = torch.nn.Linear(50, 38)
        self.m=nn.ReLU()
        self.DNNnew= nn.Sequential(
            nn.Linear(n_hidden_in, n_hidden_in),
            nn.ReLU(),
            nn.Linear(n_hidden_in, n_hidden_in),
            nn.ReLU(),
            nn.Linear(n_hidden_in, n_hidden_in),
            nn.ReLU(),
            nn.Linear(n_hidden_in, n_hidden_in),
            nn.ReLU(),
            nn.Linear(n_hidden_in, 30)
        )

    def forward(self, x_in):
        x_in = x_in[None, :, :]
        x_1, _ = self.lstm1(x_in)
        x_2, _ = self.lstm2(x_1)
        x_3, _ = self.lstm3(x_2)
        outputs_LSTM = self.fc(x_3)

        x_DNN_in = outputs_LSTM[0, :, :]
        x_DNN_in = self.m(x_DNN_in)
        x_DNN_in1 = self.l_x2(x_DNN_in)
        outputs_DNN = self.DNNnew(x_DNN_in1)

        x_2_in = x_2[0, :, :]
        x_3_in = x_2_in[:, :, None]
        x1 = self.l_x(x_3_in)
        x2 = self.c_x(x1)
        z_h = self.z_h(x2)
        alpha_out = F.softmax(self.z_alpha(z_h), -1)
        mu_out = self.z_mu(z_h)
        sigma_out = self.z_sigma(z_h) + 1.000001
        return alpha_out, mu_out, sigma_out, outputs_DNN


n_hidden = 500
n_gaussian = 1
c_x_s = int(np.floor((n_hidden - 3) / (n_hidden / 4 - 1)))
model = MDN(n_hidden_in=n_hidden, n_gaussian_in=n_gaussian, stride=c_x_s).to(device)
model.to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())


# cPNN-MDN loss define
def mdn_loss_fn(y_in, mu_in, sigma_in, alpha_in):
    y_in = y_in[:, :, None]
    m = torch.distributions.Normal(loc=mu_in, scale=sigma_in)
    loss_out = torch.exp(m.log_prob(y_in))
    loss_out = torch.sum(loss_out * alpha_in, dim=-1)
    loss_out = -torch.log(loss_out + 0.000001)
    return torch.mean(loss_out)


# Train
epoch_max = 10000
loss_data = []
loss1_data = []
loss2_data = []
loss3_data = []
early_stopping = EarlyStopping(patience=80)
#model.load_state_dict(torch.load("modelbest.pth"))
for epoch in range(epoch_max):
    model.train()
    for batch, (X, Y) in enumerate(train_dataloader):
        X, Y = X.to(device), Y.to(device)
        alpha, mu, sigma,  outputs_DNN = model(X)
        loss1 = mdn_loss_fn(Y, mu, sigma, alpha)
        loss3 = loss_fn(outputs_DNN, Y)
        loss = loss1+ loss3*10
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch =', epoch, '/', epoch_max, ',', 'loss =', loss.data.tolist(),'loss1 =', loss1.data.tolist(),'loss3 =', loss3.data.tolist())
    loss_data.append(loss.data.cpu().detach().numpy())
    loss1_data.append(loss1.data.cpu().detach().numpy())
    loss3_data.append(loss3.data.cpu().detach().numpy())
    early_stopping(loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

scipy.io.savemat('loss_h{}g{}.mat'.format(n_hidden, n_gaussian), {'loss': loss_data})
scipy.io.savemat('loss1_h{}g{}.mat'.format(n_hidden, n_gaussian), {'loss1': loss1_data})
scipy.io.savemat('loss3_h{}g{}.mat'.format(n_hidden, n_gaussian), {'loss3': loss3_data})
plt.figure(figsize=(8, 8))  # loss figure
plt.plot(loss_data)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('n_hidden = {} , n_gaussian = {}'.format(n_hidden, n_gaussian))
plt.show()

# Template for test
n_samples = 10
x_test = x_data[:n_samples, :]

model.load_state_dict(torch.load("modelbest.pth"))
x_test = x_test.to(device)
alpha, mu, sigma, outputs_DNN = model(x_test)

alpha = alpha.cpu().detach().numpy()
mu = mu.cpu().detach().numpy()
sigma = sigma.cpu().detach().numpy()
outputs_DNN = outputs_DNN.cpu().detach().numpy()
scipy.io.savemat('alpha_h{}g{}.mat'.format(n_hidden, n_gaussian), {'alpha': alpha})
scipy.io.savemat('mu_h{}g{}.mat'.format(n_hidden, n_gaussian), {'mu': mu})
scipy.io.savemat('sigma_h{}g{}.mat'.format(n_hidden, n_gaussian), {'sigma': sigma})
scipy.io.savemat('outputs_DNN_h{}g{}.mat'.format(n_hidden, n_gaussian), {'outputs_DNN': outputs_DNN})

