import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Directories
train_dir = r'C:\Users\thars\OneDrive\Desktop\project\data\train'
test_dir = r'C:\Users\thars\OneDrive\Desktop\project\data\test'
train_csvpath = r'C:\Users\thars\OneDrive\Desktop\project\data\reid_list_train.csv'
test_csvpath = r'C:\Users\thars\OneDrive\Desktop\project\data\reid_list_test.csv'


df = pd.read_csv(train_csvpath)
df.columns = ['labels', 'filepaths']
df['filepaths'] = df['filepaths'].apply(lambda x: os.path.join(train_dir, x))
df['labels'] = df['labels'].apply(lambda x: str(x))

# Split data
train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123, stratify=df['labels'])
valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123, stratify=dummy_df['labels'])

# Data generators
img_size = (200, 250)
batch_size = 30

trgen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2)
t_and_v_gen = ImageDataGenerator()

train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
valid_gen = t_and_v_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)

length = len(test_df)
test_batch_size = sorted([int(length/n) for n in range(1, length+1) if length % n == 0 and length/n <= 80], reverse=True)[0]
test_steps = int(length/test_batch_size)

test_gen = t_and_v_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)
