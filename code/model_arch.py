# V18 — 4-encoder autoencoder. dropout=0.25, latent=600.
import torch, torch.nn as nn
class Autoencoder(nn.Module):
    def __init__(self, input_dim, s1, s2, s3, s4, dropout):
        super().__init__()
        self.encoder_1 = nn.Sequential(
            nn.Linear(s1,200), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(200,100),nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(100,50), nn.ReLU(True))
        self.encoder_2 = nn.Sequential(
            nn.Linear(s2,4000),nn.ReLU(True),nn.Dropout(dropout),
            nn.Linear(4000,2000),nn.ReLU(True),nn.Dropout(dropout),
            nn.Linear(2000,1000),nn.ReLU(True),nn.Dropout(dropout),
            nn.Linear(1000,150),nn.ReLU(True))
        self.encoder_3 = nn.Sequential(
            nn.Linear(s3,4000),nn.ReLU(True),nn.Dropout(dropout),
            nn.Linear(4000,2000),nn.ReLU(True),nn.Dropout(dropout),
            nn.Linear(2000,1000),nn.ReLU(True),nn.Dropout(dropout),
            nn.Linear(1000,200),nn.ReLU(True))
        self.encoder_4 = nn.Sequential(
            nn.Linear(s4,8000),nn.ReLU(True),nn.Dropout(dropout),
            nn.Linear(8000,4000),nn.ReLU(True),nn.Dropout(dropout),
            nn.Linear(4000,2000),nn.ReLU(True),nn.Dropout(dropout),
            nn.Linear(2000,1000),nn.ReLU(True),nn.Dropout(dropout),
            nn.Linear(1000,200),nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(600,1000), nn.ReLU(True),nn.Dropout(dropout),
            nn.Linear(1000,2000),nn.ReLU(True),nn.Dropout(dropout),
            nn.Linear(2000,4000),nn.ReLU(True),nn.Dropout(dropout),
            nn.Linear(4000,8000),nn.ReLU(True),nn.Dropout(dropout),
            nn.Linear(8000,input_dim),nn.Sigmoid())
    def encode(self,x1,x2,x3,x4):
        return torch.cat([self.encoder_1(x1),self.encoder_2(x2),
                          self.encoder_3(x3),self.encoder_4(x4)], dim=1)
    def forward(self,x1,x2,x3,x4):
        return self.decoder(self.encode(x1,x2,x3,x4))
