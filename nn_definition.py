import torch.nn as nn

CLASSES = 2
DROP_OUT = 0.5
DIMENSION = 613


class TextualSpeakerChangeDetection(nn.Module):
    def __init__(self):
        super().__init__()
        self.Input_to_Hidden1 = nn.Linear(DIMENSION, int(DIMENSION/2.0) + 1)
        self.DropOut1 = nn.Dropout(p=DROP_OUT)
        self.Hidden1_to_Hidden2 = nn.Linear(int(DIMENSION/2.0) + 1, int(DIMENSION/4.0) + 1)
        self.DropOut2 = nn.Dropout(p=DROP_OUT)
        self.Hidden2_to_Hidden3 = nn.Linear(int(DIMENSION/4.0) + 1, int(DIMENSION/8.0) + 1)
        self.DropOut3 = nn.Dropout(p=DROP_OUT)
        self.Hidden3_to_Output = nn.Linear(int(DIMENSION/8.0) + 1, CLASSES)

    def forward(self, X):
        x_inp = nn.ReLU()(self.Input_to_Hidden1(X))
        x_d1 = self.DropOut1(x_inp)
        x_hid1_2 = nn.ReLU()(self.Hidden1_to_Hidden2(x_d1))
        x_d2 = self.DropOut2(x_hid1_2)
        x_hid2_3 = nn.ReLU()(self.Hidden2_to_Hidden3(x_d2))
        x_d3 = self.DropOut3(x_hid2_3)
        x_out = nn.ReLU()(self.Hidden3_to_Output(x_d3))
        y = nn.Softmax()(x_out)
        return y


    def get_steps(self):
        return 1000

    def get_learning_rate(self):
        return 0.0001

    def to_string(self):
        return "TSCD_Model-steps_"
