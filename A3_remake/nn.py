import tensorflow as tf
import numpy as np

# Definizione della rete neurale per il costo terminale
class TerminalCostNN(tf.keras.Model):
    def __init__(self):
        super(TerminalCostNN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

# Creazione del modello della rete neurale
terminal_cost_model = TerminalCostNN()

# Definizione della funzione di costo per l'apprendimento
def terminal_cost_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Creazione dei dati di addestramento
num_samples = 1000
state_samples = np.random.rand(num_samples, 4)  # Stati casuali
action_samples = np.random.rand(num_samples, 2)  # Azioni casuali
x_train = tf.constant(np.concatenate([state_samples, action_samples], axis=1), dtype=tf.float32)
y_train = tf.constant(np.random.rand(num_samples, 1), dtype=tf.float32)  # Costi casuali

# Compilazione del modello e addestramento
terminal_cost_model.compile(optimizer='adam', loss=terminal_cost_loss)
terminal_cost_model.fit(x_train, y_train, epochs=50, batch_size=32)

# Salvataggio del modello addestrato
terminal_cost_model.save('terminal_cost_model')