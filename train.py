import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
from First_Paper.TSCPD.Git_Upload.nn_definition import TextualSpeakerChangeDetection


CLASSES = 2
LABEL_INDEX = 614
BATCH_SIZE = 5000
TRAIN_VECTORS_PATH = 'train_vectors.pkl'


def get_classes(l):

    class_dict = {}
    l = sorted(list(set(list([str(x) for x in l]))))

    for i in range(len(l)):
        class_dict.update({l[i]:i})

    return class_dict


def get_tensors_x_y(x_t, y_t, cls):
    data_x = []
    data_y = []

    indices = x_t.index.values

    idx = 0
    for index in indices:
        idx += 1
        data_x.append(list(x_t.loc[index]))
        data_y.append(cls[y_t[index]])

    torch_tensor_X = torch.from_numpy(np.array(data_x))
    torch_tensor_Y = torch.from_numpy(np.array(data_y))

    return torch_tensor_X, torch_tensor_Y


train_vectors = pd.read_pickle(TRAIN_VECTORS_PATH)

X_train = train_vectors.iloc[:, :-2]
Y_train = train_vectors[LABEL_INDEX]

classes = get_classes(Y_train)

pd.to_pickle(classes, 'TSCD_classes_for_test.pkl')

print("Getting tensors")
X, Y = get_tensors_x_y(X_train, Y_train, classes)
print("Done getting tensors")

X = X.float()
Y = Y.long()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = TextualSpeakerChangeDetection().to(device)
learning_rate = model.get_learning_rate()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
steps = model.get_steps()

weights = []
for cls in sorted(classes.keys()):
    weights.append(1.0 / len(train_vectors[train_vectors[LABEL_INDEX] == cls]))
class_weights = torch.FloatTensor(weights).cuda()

criterion = nn.CrossEntropyLoss(weight=class_weights)
batch_number = int(len(train_vectors) / BATCH_SIZE) + 1


for i in range(steps):
    model.train()
    optimizer.zero_grad()

    mod = i % batch_number
    if (batch_number-1) == mod:
        index_batch_min = mod * BATCH_SIZE
        index_batch_max = len(X) - 1
    else:
        index_batch_min = mod * BATCH_SIZE
        index_batch_max = (mod + 1) * BATCH_SIZE

    x_to_device = X[index_batch_min:index_batch_max]
    y_to_device = Y[index_batch_min:index_batch_max]

    y_ = model(x_to_device.to(device))
    loss = criterion(y_, y_to_device.to(device))
    loss.backward()
    optimizer.step()

    print("Step: " + str(i+1) + ",  Loss Function: " + str(loss) +
          ",  From: " + str(index_batch_min) + ",  to: " + str(index_batch_max))

    if 0 == (i+1) % 200:
        torch.save(model, 'Models\\' + model.to_string() + str(i + 1) + ".pth")
