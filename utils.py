import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    'soil_ph', 'soil_moisture', 'temperature', 'rain_last_7d', 'day_of_year', 'previous_crop_index'
]

CROPS = ['maize', 'rice', 'cotton']

def gen_synthetic_data(num_samples=200, seed=0):
    np.random.seed(seed)
    soil_ph = np.random.normal(6.5, 0.6, num_samples)
    soil_moist = np.clip(np.random.normal(0.35, 0.12, num_samples), 0, 1)
    temp = np.random.normal(28, 5, num_samples)
    rain = np.clip(np.random.exponential(2, num_samples), 0, 20)
    day = np.random.randint(1, 366, num_samples)
    prev = np.random.randint(0, 3, num_samples)

    label = []
    for i in range(num_samples):
        if soil_ph[i] > 6.8 and soil_moist[i] > 0.3 and temp[i] > 25:
            label.append(0)  # maize
        elif rain[i] > 5 and temp[i] > 22:
            label.append(1)  # rice
        else:
            label.append(2)  # cotton

    df = pd.DataFrame({
        'soil_ph': soil_ph,
        'soil_moisture': soil_moist,
        'temperature': temp,
        'rain_last_7d': rain,
        'day_of_year': day,
        'previous_crop_index': prev,
        'label': label
    })

    return df

def prepare_data(df, test_size=0.2):
    X = df[FEATURES].values.astype('float32')
    y = df['label'].values.astype('int64')
    return train_test_split(X, y, test_size=test_size, random_state=42)
