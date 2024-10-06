import torch
import torch.nn as nn

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(129, 7), stride=7, padding=0),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(129, 7), stride=7, padding=0, output_padding=0),  
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class SeismicEventPredictor(nn.Module):
    def __init__(self, autoencoder):
        super(SeismicEventPredictor, self).__init__()
        self.encoder = autoencoder.encoder
        self.fc1 = nn.Linear(64 * 92, 64 * 92 * 4)  # Adjust dimensions based on encoder output
        self.fc2 = nn.Linear(64 * 92 * 4, )  # Adjust dimensions based on encoder output
        self.fc = nn.Linear(64 * 92, 1)  # Adjust dimensions based on encoder output
        self.activation = nn.ReLU()

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)  # Flatten
        # x = self.activation(self.fc1(encoded))
        # output = self.fc2(x)
        output = self.fc(x)
        return output