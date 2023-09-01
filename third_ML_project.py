# LINK TO VIEW CODE IN GOOGLE COLAB:
# https://colab.research.google.com/drive/14V0FJtgZb86H9JRtQAVUfsadCbin0-16?usp=sharing

import torch
import numpy as np
from torch import nn

#Inicializando uma rede neural com a biblioteca Module
#A biblioteca Module é a raiz de todas as redes neurais do Pytorch
#LineNetwork se refere a uma rede linear
class LineNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(1,1)
        #Linear(1,1) significa que será uma rede com 1 camada de neurônios
        #Um neurônio representa uma função
    )

  def forward(self, x):
   return self.layers(x)

from torch.utils.data import Dataset, DataLoader
import torch.distributions.uniform as urand #serve para aleatorizar números

#Construindo um dataset para servir como base de dados para o neurônio (a*x + b)
class AlgebraicDataset(Dataset):
  #função que irá construir um dataset recebendo os parâmetros abaixo:
  def __init__(self, f, interval, nsamples):
    #interval: intervalo de números
    #nsamples: quantidade de pontos a serem plotados
    X = urand.Uniform(interval[0], interval[1]).sample([nsamples])
    self.data = [(x, f(x)) for x in X]

  #função para saber quantos dados tem no dataset
  def __len__(self):
    return len(self.data)

  #função para buscar um determinado valor
  def __getitem__(self, index):
    return self.data[index]

line = lambda x: 2*x + 3
interval = (-10,10)
train_nsamples = 1000
test_nsamples = 100

train_dataset = AlgebraicDataset(line, interval, train_nsamples)
test_dataset = AlgebraicDataset(line, interval, test_nsamples)

train_dataloader = DataLoader(train_dataset, batch_size=train_nsamples, shuffle=True)
#serve para carregar dados provenientes do train_dataset com fluxo (batch_size) único
#o batch_size poderia ser menor, provendo "doses homeopáticas" de dados

test_dataloader = DataLoader(test_dataset, batch_size=test_nsamples, shuffle=True)
#shuffle serve para embaralhar os dados

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Rodando na {device}")
model = LineNetwork().to(device)

lossfunc = nn.MSELoss()
#MSE significa Mean Squared Error
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) #lr é a taxa de aprendizado da rede neural
#SGD signfica Stochastic Gradient Descent

def train(model, dataloader, lossfunc, optimizer):
  model.train()
  cumloss = 0.0
  for X, y in dataloader:
    X = X.unsqueeze(1).float().to(device)
    y = y.unsqueeze(1).float().to(device)
    #unsqueeze serve para transformar os elementos de um vetor em tensores
    #.float() transforma a entrada y em float de 32 bits

    pred = model(X)
    loss = lossfunc(pred, y)

    optimizer.zero_grad()
    #o PyTorch acumula os valores de cada cálculo de gradiente. "zero_grad()" serve para zerar o gradiente para calcular um novo
    loss.backward()
    #.backward() calcula o gradiente
    optimizer.step()
    #.step serve para andar na sentido contrário do gradiente, ou seja, para onde há menor erro

    #loss é um tensor que obtem o float
    cumloss += loss.item()

  return cumloss / len(dataloader)

def test(model, dataloader, lossfunc):
  model.eval()
  cumloss = 0.0

  with torch.no_grad():
  #torch.no_grad serve para acumular gradientes
    for X, y in dataloader:
      X = X.unsqueeze(1).float().to(device)
      y = y.unsqueeze(1).float().to(device)
      #unsqueeze serve para transformar os elementos de um vetor em tensores
      #.float() transforma a entrada y em float de 32 bits

      pred = model(X)
      loss = lossfunc(pred, y)

      #loss é um tensor que obtem o float
      cumloss += loss.item()

  return cumloss / len(dataloader)

def plot_comparinson(f, model, interval=(-10, 10), nsamples=10):
  fig, ax = plt.subplots(figsize=(10, 10))

  ax.grid(True, which='both')
  ax.spines['left'].set_position('zero')
  ax.spines['right'].set_color('none')
  ax.spines['bottom'].set_position('zero')
  ax.spines['top'].set_color('none')

  samples = np.linspace(interval[0], interval[1], nsamples)
  model.eval()
  with torch.no_grad():
    pred = model(torch.tensor(samples).unsqueeze(1).float().to(device))

  ax.plot(samples, list(map(f, samples)), "o", label="ground truth")
  ax.plot(samples, pred.cpu(), label="model")
  plt.legend()
  plt.show()

import matplotlib.pyplot as plt

#epochs serve para dizer quantas vezes irá rodar o treinamento
epochs = 301
for t in range(epochs):
  train_loss = train(model, train_dataloader, lossfunc, optimizer)
  if t % 10 == 0:
    print(f"Epoch: {t}; Train Loss: {train_loss}")
    plot_comparinson(line, model)

test_loss = test(model, test_dataloader, lossfunc)
print(f"Test Loss: {test_loss}")
