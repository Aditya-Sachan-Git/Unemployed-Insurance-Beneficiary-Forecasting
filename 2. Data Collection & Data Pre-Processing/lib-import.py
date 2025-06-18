import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px 
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from sklearn.metrics import mean_squared_error,mean_absolute_error,median_absolute_error,r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM 
from keras.layers import Dense
#from keras.preprocessing.sequence import TimeseriesGenerator (Incurred an error in this )
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import prophet 
from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import AutoReg
df = pd.read_csv(r'C:\Users\Sri charan\Desktop\temp\pratcise\unemployment-insurance-beneficiaries-and-benefit-amounts-paid-beginning-2001-1 (1).csv')
df.head()
df.info()
df.shape
print(df.isna().sum())
print(df.Region.value_counts())
print(df.duplicated().sum())