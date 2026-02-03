import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler

# Constants
# Adjust paths to match the user's workspace structure
# Workspace root: c:\Workspace\PYTHONN\Predicting-Segment-Value-Of-House-In-TPHCM
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "house_segment_rf_optimized.pkl")
# Use the None_Scalling file to get the schema and fit the scaler
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned", "Model_None_Scaling.csv")

st.set_page_config(page_title="Dự đoán Phân khúc Nhà", layout="centered")

@st.cache_resource
def load_resources():
    try:
        # Load reference data to fit scaler and validation schema
        if not os.path.exists(DATA_PATH):
            st.error(f"Không tìm thấy file dữ liệu tại: {DATA_PATH}")
            return None, None, None, None

        df = pd.read_csv(DATA_PATH)
        
        # Define numerical columns that need scaling (as per user logic)
        num_cols = ['Area_m2', 'Width_m', 'Floors', 'Bedrooms', 'Toilets']
        
        # Fit scaler
        scaler = StandardScaler()
        # Ensure only numeric types are selected just in case, though they should be numeric
        scaler.fit(df[num_cols])
        
        # Load model
        if not os.path.exists(MODEL_PATH):
            st.error(f"Không tìm thấy file model tại: {MODEL_PATH}")
            return None, None, None, None

        # Try loading with joblib first (standard for sklearn), then pickle
        import joblib
        try:
            model = joblib.load(MODEL_PATH)
        except:
             with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            
        return scaler, model, df.columns.tolist(), df.dtypes
    except Exception as e:
        st.error(f"Lỗi khi tải tài nguyên: {str(e)}")
        return None, None, None, None

scaler, model, all_columns, column_dtypes = load_resources()

if scaler and model:
    st.title("🏡 Dự đoán Phân khúc Nhà Đất TP.HCM")
    st.markdown("---")

    # Mapping target to class names
    # Based on user report: ['Bình dân', 'Trung cấp', 'Cao cấp', 'VIP']
    # Usually encoded as 0, 1, 2, 3
    segment_map = {0: 'Bình dân', 1: 'Trung cấp', 2: 'Cao cấp', 3: 'VIP'}

    # Prepare list of districts from columns
    # Expecting columns like 'Dist_Bình Thạnh', 'Dist_Quận 1', etc.
    dist_cols = [c for c in all_columns if c.startswith('Dist_')]
    dist_names = [c.replace('Dist_', '') for c in dist_cols]
    dist_names.sort()

    with st.form("prediction_form"):
        st.subheader("Thông tin chi tiết")
        
        col1, col2 = st.columns(2)
        
        with col1:
            area = st.number_input("Diện tích đất (m²)", min_value=1.0, value=50.0, step=1.0)
            width = st.number_input("Chiều rộng (m)", min_value=1.0, value=4.0, step=0.5)
            floors = st.number_input("Số tầng", min_value=1.0, value=2.0, step=1.0)
            price_billion = st.number_input("Giá bán (Tỷ VND)", min_value=0.1, value=5.0, step=0.1)

        with col2:
            bedrooms = st.number_input("Số phòng ngủ", min_value=0, value=2, step=1)
            toilets = st.number_input("Số toilet", min_value=0, value=2, step=1)
            position = st.radio("Vị trí nhà", ["Mặt Tiền", "Hẻm"], horizontal=True)
            district = st.selectbox("Quận / Huyện", dist_names)

        # Confirm calculation
        price_per_m2_million = (price_billion * 1000) / area if area > 0 else 0
        st.info(f"Đơn giá tính toán: {price_per_m2_million:,.2f} Triệu/m²")

        submit = st.form_submit_button("Dự đoán Phân khúc", use_container_width=True)

    if submit:
        # Create input dict (single row)
        # Initialize all columns with 0
        input_row = {col: 0 for col in all_columns if col != 'Segment_Encoded'}
        
        # Fill inputs
        input_row['Area_m2'] = float(area)
        input_row['Width_m'] = float(width)
        input_row['Floors'] = float(floors)
        input_row['Bedrooms'] = int(bedrooms)
        input_row['Toilets'] = int(toilets)
        input_row['Price_Billion'] = float(price_billion)
        input_row['Price_Per_m2'] = price_per_m2_million
        
        # Handle Boolean/Binary columns
        # Is_MatTien and Is Hem
        # Based on data sample: Is_MatTien=1, Is Hem=0 (Integers)
        is_mt = 1 if position == "Mặt Tiền" else 0
        is_h = 1 if position == "Hẻm" else 0
        
        input_row['Is_MatTien'] = is_mt
        if 'Is Hem' in input_row:
            input_row['Is Hem'] = is_h
        elif 'Is_Hem' in input_row: # Check for underscore var just in case
            input_row['Is_Hem'] = is_h
            
        # Handle District One-Hot
        selected_dist_col = f"Dist_{district}"
        if selected_dist_col in input_row:
            # Check original dtype for this column
            dtype = column_dtypes[selected_dist_col]
            if pd.api.types.is_bool_dtype(dtype):
                 input_row[selected_dist_col] = True
            else:
                 input_row[selected_dist_col] = 1

        # Create DataFrame
        df_input = pd.DataFrame([input_row])
        
        # Ensure column order matches exactly
        feature_cols = [c for c in all_columns if c != 'Segment_Encoded']
        df_input = df_input[feature_cols]

        # Fix types for Booleans if they were initialized as 0 (int) but need to be compatible
        # Generally 0/1 works for RF, but let's be safe
        # (Already handled above by checking dtype for District)

        # Scale numerical columns
        num_cols = ['Area_m2', 'Width_m', 'Floors', 'Bedrooms', 'Toilets']
        df_input[num_cols] = scaler.transform(df_input[num_cols])
        
        # Drop columns that were not used in training (likely Price info to avoid leakage)
        cols_to_drop = ['Price_Billion', 'Price_Per_m2']
        df_input = df_input.drop(columns=[c for c in cols_to_drop if c in df_input.columns])

        # Predict
        try:
            pred_class = model.predict(df_input)[0]
            pred_label = segment_map.get(pred_class, f"Unknown ({pred_class})")
            
            # Display Result
            st.success(f"Kết quả dự đoán: **{pred_label}**")
            
            # Optional: Probability
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df_input)[0]
                st.write("Độ tin cậy:")
                probs = {segment_map.get(i, i): p for i, p in enumerate(proba)}
                st.bar_chart(probs)
                
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")
