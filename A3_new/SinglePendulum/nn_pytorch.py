import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carica il dataset da un file CSV
dataset_path = 'single_data.csv'  # Sostituisci con il percorso effettivo del tuo dataset
data = pd.read_csv(dataset_path)

# Estrai colonne di input (posizione e velocità) e output (etichetta di classificazione)
X = data[['q', 'v']].values
y = data['viable'].values

# Normalizza le colonne di input
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Suddividi il dataset in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Converte i dati in tensori PyTorch
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1,1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1,1)

# Definisci la rete neurale con più layer
class PendulumClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(PendulumClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Parametri della rete con più layer
input_size = 2  # Numero di features (posizione e velocità)
hidden_size1 = 128  # Numero di neuroni nel primo layer nascosto
hidden_size2 = 64   # Numero di neuroni nel secondo layer nascosto

# Inizializza il modello, la funzione di perdita e l'ottimizzatore
model = PendulumClassifier(input_size, hidden_size1, hidden_size2)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Addestramento del modello
num_epochs = 1000

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass e ottimizzazione
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Stampa la perdita ad ogni 100 epoche
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Salvataggio del modello
torch.save(model.state_dict(), 'pendulum_model.pth')

# Valutazione del modello
with torch.no_grad():
    model.eval()
    outputs = model(X_test)
    predictions = torch.sigmoid(outputs)
    predicted_labels = (predictions >= 0.5).float()  # Converte le probabilità in etichette binarie
    accuracy = torch.sum(predicted_labels == y_test).item() / y_test.size(0)
    print(f'Accuracy on test set: {accuracy:.4f}')