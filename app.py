
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    artifact = joblib.load("Life_Expectancy_Model.pkl")

    return artifact

artifact = load_model()

model = artifact["model"]

model_name = artifact["model_name"]

feature_columns = artifact["feature_columns"]

numeric_columns = artifact["numeric_columns"]

categorical_columns = artifact["categorical_columns"]

numeric_defaults = artifact["numeric_defaults"]

numeric_min = artifact["numeric_min"]

numeric_max = artifact["numeric_max"]

category_options = artifact["category_options"]

integer_columns = artifact["integer_columns"]


# ==========================================
# JUDUL APLIKASI
# ==========================================

st.set_page_config(
    page_title="Prediksi Life Expectancy",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Prediksi Life Expectancy")

st.write(
    "Aplikasi ini digunakan untuk memprediksi "
    "angka harapan hidup berdasarkan indikator "
    "kesehatan, ekonomi, dan pendidikan suatu negara."
)

st.caption(
    f"Model Machine Learning yang digunakan: {model_name}"
)

st.divider()


# ==========================================
# INPUT PARAMETER
# ==========================================

st.sidebar.header("Input Parameter")

st.sidebar.write(
    "Masukkan nilai indikator untuk melakukan prediksi."
)

input_data = {}

with st.sidebar:

    st.subheader("Indikator Numerik")

    for col in numeric_columns:

        min_val = numeric_min[col]
        max_val = numeric_max[col]
        default_val = numeric_defaults[col]

        if min_val == max_val:
            min_val -= 1
            max_val += 1

        default_val = max(
            min_val,
            min(default_val, max_val)
        )

        # Input untuk variabel integer
        if col in integer_columns:

            min_val = int(np.floor(min_val))
            max_val = int(np.ceil(max_val))
            default_val = int(round(default_val))

            if min_val == max_val:
                min_val -= 1
                max_val += 1

            input_data[col] = st.number_input(
                col,
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=1
            )

        # Input untuk variabel desimal
        else:

            input_data[col] = st.number_input(
                col,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=0.1,
                format="%.4f"
            )

    st.divider()

    st.subheader("Indikator Kategorikal")

    for col in categorical_columns:

        options = category_options.get(
            col,
            []
        )

        if len(options) == 0:
            options = ["Unknown"]

        input_data[col] = st.selectbox(
            col,
            options
        )


# ==========================================
# MENYIAPKAN DATA INPUT
# ==========================================

df_input = pd.DataFrame(
    [input_data],
    columns=feature_columns
)


# ==========================================
# MENAMPILKAN INPUT PENGGUNA
# ==========================================

st.subheader("Parameter Input Pengguna")

st.dataframe(
    df_input,
    use_container_width=True
)


# ==========================================
# MELAKUKAN PREDIKSI
# ==========================================

if st.sidebar.button(
    "Prediksi Life Expectancy",
    type="primary"
):

    try:

        prediction = model.predict(
            df_input
        )

        hasil = prediction[0]

        st.divider()

        st.subheader(
            "Hasil Prediksi Life Expectancy"
        )

        st.metric(
            label="Prediksi Angka Harapan Hidup",
            value=f"{hasil:.2f} Tahun"
        )

        st.success(
            "Prediksi berhasil dilakukan!"
        )

        st.write(
            f"Model yang digunakan: {model_name}"
        )

    except Exception as e:

        st.error(
            f"Terjadi kesalahan saat prediksi: {e}"
        )

        st.exception(e)


# ==========================================
# INFORMASI APLIKASI
# ==========================================

st.divider()

st.caption(
    "Aplikasi Prediksi Life Expectancy "
    "menggunakan Machine Learning."
)

st.caption(
    "Hasil prediksi merupakan estimasi model "
    "berdasarkan pola data pelatihan, bukan "
    "angka resmi suatu negara."
)
