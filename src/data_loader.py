import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Loads the csv data."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Scales data and splits it into inputs and targets."""
    mms = MinMaxScaler()
    df_scaled = df.copy()
    for i in df_scaled.columns:
        df_scaled[i] = mms.fit_transform(df_scaled[[i]])
    
    inputs = df_scaled.iloc[:, :-1]
    target = df_scaled.iloc[:, -1]
    
    return inputs, target

def get_train_test_split(inputs, target, test_size=0.25):
    """Splits data into training and testing sets."""
    return train_test_split(inputs, target, test_size=test_size)
