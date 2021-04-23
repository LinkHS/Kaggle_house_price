from torch import nn

def model_v1_1(in_ch):
    dropout1 = 0.2
    dropout2 = 0.2
    model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_ch, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(dropout1),
              nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout2),
              nn.Linear(256, 1))
    return model

def model_v1_2(in_ch):
    dropout1 = 0.0
    dropout2 = 0.0
    dropout3 = 0.0
    model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_ch, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(dropout1),
              nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(dropout2),
              nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout3),
              nn.Linear(256, 1))    
    return model

def model_v1_3(in_ch):
    dropout1 = 0.2
    dropout2 = 0.2
    model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_ch, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout1),
              nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout2),
              nn.Linear(256, 1))    
    return model


def model_v1_4(in_ch):
    dropout1 = 0.2
    dropout2 = 0.2
    model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_ch, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout1),
              nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout2),
              nn.Linear(256, 1))    
    return model


def model_v1_5(in_ch):
    dropout1 = 0.2
    dropout2 = 0.2
    model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_ch, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout1),
              nn.Linear(512, 16), nn.ReLU(), nn.Dropout(dropout2),
              nn.Linear(16, 1))    
    return model


def model_v1_6(in_ch):
    dropout1 = 0
    dropout2 = 0
    model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_ch, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout1),
              nn.Linear(256, 1))
    return model


def model_v1_7(in_ch):
    dropout1 = 0.2
    model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_ch, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout1),
              nn.Linear(256, 1))
    return model


def model_v1_8(in_ch):
    dropout1 = 0.2
    dropout2 = 0.2
    model = nn.Sequential(nn.Flatten(),
            nn.Linear(in_ch, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout1),
            nn.Linear(256, 16), nn.ReLU(), nn.Dropout(dropout2),
            nn.Linear(16, 1))
    return model


def model_v1_9(in_ch):
    dropout1 = 0.2
    dropout2 = 0.2
    model = nn.Sequential(nn.Flatten(),
            nn.Linear(in_ch, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout2),
            nn.Linear(128, 1))
    return model


def model_v2_1(in_ch):
    dropout1 = 0.2
    dropout2 = 0.2
    dropout3 = 0.2
    model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_ch, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(dropout1),
              nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(dropout2),
              nn.Linear(512, 512), nn.ReLU(),  nn.Dropout(dropout3),
              nn.Linear(512, 1))    
    return model


def model_v2_2(in_ch):
    dropout1 = 0.2
    dropout2 = 0.2
    dropout3 = 0.2
    model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_ch, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(dropout1),
              nn.Linear(1024, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(dropout2),
              nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(dropout2),
              nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout3),
              nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout3),
              nn.Linear(256, 1))    
    return model



def model_v3_1(in_ch):
    dropout1 = 0.0
    dropout2 = 0.0
    dropout3 = 0.0
    model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_ch, 2048), nn.ReLU(), nn.BatchNorm1d(2048), nn.Dropout(dropout1),
              nn.Linear(2048, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(dropout2),
              nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(dropout3),
              nn.Linear(512, 1))    
    return model


def model_v3_2(in_ch):
    dropout1 = 0.2
    dropout2 = 0.2
    dropout3 = 0.2
    dropout4 = 0.2
    model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_ch, 2048), nn.ReLU(), nn.BatchNorm1d(2048), nn.Dropout(dropout1),
              nn.Linear(2048, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(dropout2),
              nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(dropout3),
              nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout4),
              nn.Linear(256, 1))    
    return model


def model_v3_3(in_ch):
    dropout1 = 0.2
    dropout2 = 0.2
    dropout3 = 0.2
    dropout4 = 0.2
    model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_ch, 2048), nn.ReLU(), nn.BatchNorm1d(2048), nn.Dropout(dropout1),
              nn.Linear(2048, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(dropout2),
              nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(dropout3),
              nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout4),
              nn.Linear(512, 1))    
    return model



def model_v4_1(in_ch):
    dropout1 = 0.2
    dropout2 = 0.2
    model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_ch, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout1),
              nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout2),
              nn.Linear(32, 1))
    return model


def model_v4_2(in_ch):
    dropout1 = 0.2
    dropout2 = 0.2
    model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_ch, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout1),
              nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout2),
              nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout2),
              nn.Linear(16, 1))
    return model


def model_v4_3(in_ch):
    dropout1 = 0.2
    dropout2 = 0.2
    model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_ch, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout1),
              nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout2),
              nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout2),
              nn.Linear(16, 8), nn.ReLU(), nn.Dropout(dropout2),
              nn.Linear(8, 1))
    return model

def model_v4_4(in_ch):
    dropout1 = 0.2
    dropout2 = 0.2
    model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_ch, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout1),
              nn.Linear(64, 32), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout2),
              nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout2),
              nn.Linear(16, 8), nn.ReLU(), nn.Dropout(dropout2),
              nn.Linear(8, 1))
    return model


def model_v5_1(in_ch):
    dropout1 = 0.2
    dropout2 = 0.2
    model = nn.Sequential(nn.Flatten(),
              nn.Linear(in_ch, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout1),
              nn.Linear(32, 8), nn.ReLU(), nn.Dropout(dropout2),
              nn.Linear(8, 1))
    return model