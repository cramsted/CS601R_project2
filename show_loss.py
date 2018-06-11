import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('losses4.pkl', 'rb') as f:
    losses = pickle.load(f)
    train_loss = losses[0]
    test_loss = losses[1]

# import pdb
# pdb.set_trace()
plt.subplot(121)
plt.plot(np.arange(0, 25, 25/len(train_loss)), train_loss, color='blue')
plt.title("Training Loss per epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.subplot(122)
plt.plot(np.arange(0, 25, 25/len(test_loss)), test_loss, color='orange')
plt.title("Test Loss per epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
