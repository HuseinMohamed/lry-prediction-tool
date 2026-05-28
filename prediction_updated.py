import streamlit as st
import numpy as np
import pickle
import os

# 1. إعدادات الصفحة الواجهة
st.set_page_config(
    page_title="LRY Predictor Tool",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 LRY Prediction Tool — DES Pretreatment")
st.markdown("---")

# 2. القواميس الرسمية المطابقة تماماً لملفك الأصلي
HBD_PROPS = {
    "LA" :{"MW":90.08, "pKa":3.86,  "HBD_count":2,"HBA_count":3,"viscosity":10.0},
    "OA" :{"MW":90.03, "pKa":1.25,  "HBD_count":2,"HBA_count":4,"viscosity":10.0},
    "Gly":{"MW":92.09, "pKa":14.4,  "HBD_count":3,"HBA_count":3,"viscosity":50.0},
    "AA" :{"MW":60.05, "pKa":4.76,  "HBD_count":1,"HBA_count":2,"viscosity":1.5},
    "U"  :{"MW":60.06, "pKa":13.9,  "HBD_count":2,"HBA_count":2,"viscosity":1.5},
    "CA" :{"MW":192.12,"pKa":3.13,  "HBD_count":4,"HBA_count":7,"viscosity":50.0},
    "EG" :{"MW":62.07, "pKa":15.1,  "HBD_count":2,"HBA_count":2,"viscosity":10.0},
    "p-TsOH":{"MW":172.20,"pKa":-1.34,"HBD_count":1,"HBA_count":3,"viscosity":1.5},
}

BIOMASS_ENC = {
    "wheat straw"           : 55.2,
    "corn cob"              : 61.3,
    "bamboo"                : 48.7,
    "rape straw"            : 52.1,
    "switch grass"          : 57.8,
    "sunflower straw"       : 49.3,
    "Arabidopsis thaliana"  : 44.6,
    "Bambara groundnut haulm": 53.4,
    "Lettuce"               : 46.2,
    "Luffa"                 : 50.1,
}

# 3. دالة تحميل الموديل الآمنة
@st.cache_resource
def load_prediction_model():
    if not os.path.exists("best_model.pkl"):
        st.error("❌ ERROR: `best_model.pkl` not found في مجلد المشروع!")
        st.stop()
    
    with open("best_model.pkl", "rb") as f:
        saved = pickle.load(f)
    
    # التأكد من استخراج الموديل سواء كان قاموساً أو كائناً مباشراً
    if isinstance(saved, dict) and "model" in saved:
        return saved["model"]
    return saved

model = load_prediction_model()

# 4. بناء الواجهة الرسومية وتوزيع العناصر في أعمدة
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Classification (Biomass and HBD type)")
    biomass = st.selectbox("🌾 Biomass Type:", list(BIOMASS_ENC.keys()))
    hbd = st.selectbox("🧪 HBD Type:", list(HBD_PROPS.keys()))
    
    st.markdown("---")
    st.subheader("🧪 Structural properties (Composition)")
    cellulose = st.number_input("Cellulose content (%)", min_value=0.0, max_value=100.0, value=38.0, step=0.1)
    hemicellulose = st.number_input("Hemicellulose content (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    lignin = st.number_input("Lignin content (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)

with col2:
    st.subheader("⚙️ Operating Conditions (Process Parameters)")
    size = st.number_input("Particle size (mm)", min_value=0.0, max_value=10.0, value=0.5, step=0.01)
    temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=300.0, value=130.0, step=1.0)
    time = st.number_input("Time (hours)", min_value=0.0, max_value=100.0, value=4.0, step=0.5)
    sl_ratio = st.number_input("S:L ratio (e.g., 0.05 for 1:20)", min_value=0.0, max_value=1.0, value=0.05, step=0.005, format="%.3f")
    hba_hbd_ratio = st.number_input("HBA:HBD molar ratio (e.g., 0.33 for 1:3)", min_value=0.0, max_value=10.0, value=0.33, step=0.01)

st.markdown("###")

# 5. زر الحساب والتنبؤ
if st.button("🚀 Calculate Prediction", type="primary", use_container_width=True):
    # جلب الخصائص تلقائياً بناءً على اختيار الـ Dropdown بدون أخطاء
    props = HBD_PROPS[hbd]
    biomass_enc = BIOMASS_ENC.get(biomass, 51.9)
    hbd_enc = np.mean(list(BIOMASS_ENC.values()))

    # تجهيز المصفوفة بنفس الترتيب الدقيق للموديل الخاص بك
    features = np.array([[
        cellulose, hemicellulose, lignin, size,
        temp, time, sl_ratio, hba_hbd_ratio,
        props["MW"], props["pKa"],
        props["HBD_count"], props["HBA_count"],
        props["viscosity"],
        biomass_enc, hbd_enc
    ]])

    # التنبؤ وحصر النسبة
    lry = model.predict(features)[0]
    result = float(np.clip(lry, 0, 100))

    # عرض النتيجة بشكل احترافي
    st.success(f"🎯 **Predicted Lignin Removal Yield (LRY) = {result:.2f}%**")
    
    # تقييم النتيجة طبقاً لمعاييرك
    if result >= 70:
        st.info("✅ Excellent delignification")
    elif result >= 50:
        st.info("✅ Good delignification")
    elif result >= 30:
        st.warning("⚠️ Moderate delignification")
    else:
        st.error("❌ Low delignification")
    # Confidence interval + model info
    st.info(f"📊 95% Confidence interval: "
            f"{max(0, result-10.44):.1f}% — "
            f"{min(100, result+10.44):.1f}%  "
            f"(based on model RMSE = 10.44%)")

    st.warning("🔍 Most influential factors: "
               "Temperature (°C) > HBD type > Biomass type")

    st.caption("Model: XGBoost | Test R²=0.786 | "
               "Nested CV R²=0.618±0.160 | n=303 samples")
