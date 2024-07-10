import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # Removing extra dimension
        out = self.fc(out[:, -1, :])
        return out

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size=13, hidden_size=128, num_layers=2, num_classes=5)  # Ensure num_classes matches the training model
model.load_state_dict(torch.load('lstm_model.pth', map_location=device))
model.to(device)
model.eval()

# Load label encoder classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)

# Load scaler used for normalizing features
scaler = StandardScaler()
scaler.mean_ = np.load('scaler_mean.npy')
scaler.scale_ = np.load('scaler_scale.npy')

# Function to extract and normalize MFCC features from audio
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

# Streamlit app
st.title("Sistem Speech Emotion Recognition")
st.subheader("Kelompok D2")
st.caption("Anggota Kelompok: \n"
               "1. Tun Pasek Sarwiko Dipranoto (2208561023)\n"
               "2. I Gede Yogananda Adi Baskara (2208561061)\n"
               "3. Putu Chandra Mayoni (2208561111)\n"
               "4. I Komang Dwi Prayoga (2208561117)")
st.divider()
uploaded_file = st.file_uploader('Choose an audio file', type=['wav', 'mp3'])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with open('temp_audio_file', 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Play the uploaded audio file
    st.audio(uploaded_file, format='audio/wav')

    # Extract and normalize MFCC features
    mfcc_features = extract_mfcc('temp_audio_file')
    mfcc_features = scaler.transform([mfcc_features])  # Normalize using the same scaler as during training
    mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(1).to(device)  # Adding sequence dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(mfcc_tensor)
        _, predicted = torch.max(outputs.data, 1)
        emotion = label_encoder.inverse_transform(predicted.cpu().numpy())

    st.write(f'Predicted Emotion: {emotion[0]}')