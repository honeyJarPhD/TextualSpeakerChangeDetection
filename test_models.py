import os
import glob
import torch
import xlsxwriter
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support


LABEL_INDEX = 614
TEST_VECTORS_PATH = 'test_vectors.pkl'
CLASSES_PATH = 'TSCD_classes_for_test.pkl'
RESULTS_PATH = 'C:\\Users\\orhai\\PycharmProjects\\SpeakersSeparator\\First_Paper\\TSCPD\\Git_Upload\\Results\\'


test_vectors = pd.read_pickle(TEST_VECTORS_PATH)

X_test = test_vectors.iloc[:, :-2]
Y_test = list(test_vectors[LABEL_INDEX])

classes = pd.read_pickle(CLASSES_PATH)
inverse_classes = {v: k for k, v in classes.items()}

data_x = []
indices = X_test.index.values

idx = 0
for index in indices:
    idx += 1
    data_x.append(list(X_test.loc[index]))

test_x = np.array(data_x)
torch_tensor_X = torch.from_numpy(test_x).float()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.chdir("Models\\")
all_models_name = glob.glob("*.pth")
all_models_name.sort(key=lambda x: os.path.getmtime(x))

results = []
for k in range(len(all_models_name)):

    y_pred = []
    y_true = []

    model_name = all_models_name[k]
    filename = Path("Models\\" + model_name)
    model = torch.load(filename.name, map_location=device)
    model.eval()

    inputs = torch_tensor_X.to(device)
    predictions = model.forward(inputs)

    for i in range(len(predictions)):
        prediction = predictions[i]
        prediction = [prediction.data[0].item(), prediction.data[1].item()]
        prediction = [round(x, 3) for x in prediction]

        true_class = Y_test[i]
        max_index = prediction.index(max(prediction))
        prediction_class = inverse_classes[max_index]
        y_pred.append(prediction_class)
        y_true.append(true_class)

    accuracy = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    precision = accuracy[0]
    recall = accuracy[1]
    f_score = (2 * precision * recall) / (precision + recall)

    results.append([str(model_name), round(precision * 100, 3),
                    round(recall * 100, 3), round(f_score * 100, 3)])


fileName = RESULTS_PATH + 'TSCD_Models_Results.xlsx'
workbook = xlsxwriter.Workbook(fileName)
worksheet = workbook.add_worksheet()
bold = workbook.add_format({'bold': True})
worksheet.set_column(0, 6, 15)

worksheet.write(0, 0, "Model Name", bold)
worksheet.write(0, 1, "Precision", bold)
worksheet.write(0, 2, "Recall", bold)
worksheet.write(0, 3, "F1-Score", bold)


for k in range(len(results)):
    for j in range(0, len(results[0])):
        worksheet.write((k + 1), j, results[k][j])

workbook.close()
