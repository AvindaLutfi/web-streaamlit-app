import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Fungsi untuk menghitung korelasi
def hitung_korelasi(data_x, data_y):
    correlation, _ = pearsonr(data_x, data_y)
    return correlation

# Fungsi untuk menghitung regresi
def hitung_regresi(data_x, data_y):
    reg = LinearRegression().fit(data_x.reshape(-1, 1), data_y)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    return slope, intercept

# Tampilan web menggunakan Streamlit
def main():
    st.title('Analisis Korelasi dan Regresi')

    # Input data
    st.subheader('Masukkan data')
    data_x = st.text_input('Data X (pisahkan dengan koma)')
    data_y = st.text_input('Data Y (pisahkan dengan koma)')

    if data_x and data_y:
        # Konversi data ke array numpy
        data_x = np.array(data_x.split(',')).astype(float)
        data_y = np.array(data_y.split(',')).astype(float)

        # Analisis korelasi
        correlation = hitung_korelasi(data_x, data_y)
        st.write(f'Korelasi: {correlation:.2f}')

        # Analisis regresi
        slope, intercept = hitung_regresi(data_x, data_y)
        st.write(f'Regresi: Y = {slope:.2f} * X + {intercept:.2f}')

        # Visualisasi data
        df = pd.DataFrame({'X': data_x, 'Y': data_y})
        st.subheader('Visualisasi Data')
        st.line_chart(df)

if __name__ == '__main__':
    main()
