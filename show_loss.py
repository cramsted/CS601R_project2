import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('running_loss.pkl', 'rb') as f:
    running_loss = pickle.load(f)

# import pdb
# pdb.set_trace()

plt.plot(np.arange(0, 15, 15/len(running_loss)), running_loss)
plt.title("Loss per epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
