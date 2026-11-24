import os
import tempfile
import base64
import sqlite3
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI
import streamlit as st

# KONFIGURASI
DB_NAME = os.path.join(tempfile.gettempdir(), "sotong_chat.db")
IMG_SIZES = {'effv2s': 288, 'convnext': 224}
CONFIDENCE_THRESHOLD = 0.65 

# 1. DATABASE
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (session_id TEXT, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def save_message(session_id, role, content):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
    conn.commit()
    conn.close()

def load_messages(session_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT role, content FROM chat_history WHERE session_id=? ORDER BY timestamp ASC", (session_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]

def clear_messages(session_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()

# 2. PROMPT LOADER
def load_prompt(filename, **kwargs):
    filepath = os.path.join("prompts", filename)
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read().format(**kwargs)
    except FileNotFoundError:
        return "Anda adalah asisten AI medis."

# 3. ARSITEKTUR DEEP LEARNING
def build_model(name, img, num_classes=4, pretrained=False):
    w = 'imagenet' if pretrained else None
    if name == 'effv2s':
        base = tf.keras.applications.EfficientNetV2S(include_top=False, weights=w, input_shape=(img, img, 3), include_preprocessing=True)
    else:
        base = tf.keras.applications.ConvNeXtTiny(include_top=False, weights=w, input_shape=(img, img, 3))
    inp = layers.Input((img, img, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    return models.Model(inp, out), base

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature)); in_channels = feature
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]
            if x.shape != skip.shape: x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat((skip, x), dim=1); x = self.ups[idx + 1](x)
        return torch.sigmoid(self.final_conv(x))

# 4. MODEL LOADER & PREDIKSI
@st.cache_resource(show_spinner="Memuat Model AI...")
def load_models():
    MODEL_DIR = './models'
    names = json.load(open(f'{MODEL_DIR}/class_names.json'))
    cls_models = {}
    for name, img in IMG_SIZES.items():
        m, _ = build_model(name, img, len(names), pretrained=False)
        m.load_weights(f'{MODEL_DIR}/{name}.weights.h5')
        cls_models[name] = m
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seg = UNet(in_channels=1, out_channels=1).to(device)
    seg.load_state_dict(torch.load(f'{MODEL_DIR}/best_segmentation_model.pth', map_location=device))
    seg.eval()
    return cls_models, seg, names, device

def validate_image_with_ai(client, vision_model, uploaded_file):
    uploaded_file.seek(0)
    base64_image = base64.b64encode(uploaded_file.read()).decode('utf-8')
    prompt_teks = load_prompt("vision_system.txt")
    
    response = client.chat.completions.create(
        model=vision_model,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt_teks}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
        max_tokens=10, temperature=0.0
    )
    return "YA" in response.choices[0].message.content.strip().upper()

def predict_image(uploaded_file, cls_models, model_seg, class_names, device):
    uploaded_file.seek(0)
    gray = cv2.imdecode(np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    probs = []
    for name, model in cls_models.items():
        s = IMG_SIZES[name]
        x = cv2.resize(gray, (s, s)).astype('float32')
        x = np.stack([x] * 3, axis=-1)[None, ...]
        p = model(x, training=False) + model(x[:, :, ::-1, :], training=False)
        probs.append((p / 2).numpy()[0])
    
    cls_pred = np.mean(probs, axis=0)
    k = int(np.argmax(cls_pred))
    pred_label = class_names[k]
    confidence = float(cls_pred[k])

    if confidence < CONFIDENCE_THRESHOLD:
        return "Tidak Diketahui", confidence, gray, None

    img_256 = cv2.resize(gray, (256, 256)) / 255.0
    x_tensor = torch.tensor(img_256, dtype=torch.float32)[None, None, ...].to(device)
    with torch.no_grad():
        seg_pred = (model_seg(x_tensor) > 0.5).float().cpu().squeeze().numpy()

    overlay = np.stack([img_256] * 3, axis=-1)
    overlay[seg_pred > 0.5] = [1, 0, 0]

    return pred_label, confidence, gray, overlay

# 5. API GROQ (CHAT)
def get_chat_response(client, model_name, system_context, chat_history):
    messages_for_groq = [{"role": "system", "content": system_context}] + chat_history
    stream = client.chat.completions.create(model=model_name, messages=messages_for_groq, stream=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content