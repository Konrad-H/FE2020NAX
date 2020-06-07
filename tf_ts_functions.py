import numpy as np
import matplotlib.pyplot as plt

def multivariate_data(features, target, start_index, end_index, history_size,
                      target_size, single_step = False):
  data = []
  labels = []
  start_index = start_index + history_size
  if end_index is None:
    end_index = len(features) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i+1)
    data.append(features[indices])

    if single_step:
      labels.append(target[i]) # era i+target_size
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)


def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()
