import streamlit as st
import joblib
import numpy as np

# 1. Load the "Brain" of your project
# Make sure you ran joblib.dump in Jupyter first!
model = joblib.load('tb_model.pkl')

# 2. Setup the Web Interface
st.set_page_config(page_title="TB Knowledge Predictor", layout="centered")
st.title("🩺 TB Knowledge Prediction System")
st.markdown("---")
st.write("Enter the demographic details below to predict TB transmission knowledge.")

# 3. Create the Form Questions
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        education = st.selectbox("Education Level", options=[0, 1, 2, 3], 
                                format_func=lambda x: ["No Education", "Primary", "Secondary", "Higher"][x])
        wealth = st.selectbox("Wealth Index", options=[1, 2, 3, 4, 5], 
                             format_func=lambda x: ["Poorest", "Poorer", "Middle", "Richer", "Richest"][x-1])
    
    with col2:
        residence = st.radio("Residence Type", options=[1, 2], format_func=lambda x: "Urban" if x==1 else "Rural")
        sex = st.radio("Sex", options=[1, 2], format_func=lambda x: "Male" if x==1 else "Female")

    weight = st.slider("Survey Weight (Normalized)", 0.0, 5.0, 1.0)

# 4. Prediction Logic
if st.button("Analyze Results", use_container_width=True):
    # Convert inputs to the format the model expects
    features = np.array([[weight, residence, education, wealth, sex]])
    
    # Get probability
    probability = model.predict_proba(features)[:, 1][0]
    
    st.markdown("### **Prediction Output**")
    
    # Using your "Sweet Spot" Logic (90% accuracy threshold)
    if probability >= 0.87:
        st.success("✅ **Result: High TB Knowledge Confirmed**")
        st.info("""
        **Message for the User:**
        * You have a strong understanding of how TB spreads.
        * **Fact:** You know that TB is airborne (spread through coughing/sneezing).
        * **Action:** You can help your community by encouraging others to get tested if they have a cough for more than 2 weeks.
        """)

    elif probability <= 0.13:
        st.error("🚨 **Result: Knowledge Gap Detected**")
        st.warning("""
        **Important TB Facts you should know:**
        1. **How it spreads:** TB is spread through the **AIR** when someone with the disease coughs, speaks, or sings.
        2. **How it does NOT spread:** You cannot get TB by sharing food, shaking hands, or through witchcraft.
        3. **Prevention:** Always cover your mouth when coughing and ensure good ventilation (open windows).
        """)

    else:
        st.warning("⚠️ **Result: Uncertain Knowledge Level**")
        st.write("The system cannot determine your knowledge level with 90% certainty. Please review the TB prevention guide below.")
    if probability <= 0.13:
        st.error("🚨 **Knowledge Gap Detected**")
        st.write("### **Specific Misconceptions to Address:**")
        
        # Display the specific knowledge points as a checklist for the health worker
        st.markdown("""
        * ❌ **Sexual Contact:** Clarify that TB is NOT a sexually transmitted infection.
        * ❌ **Mosquito Bites:** Explain that TB is NOT spread by insects.
        * 🧪 **Cure:** Emphasize that **TB CAN BE CURED** with a full course of antibiotics.
        * 💨 **Truth:** Reinforce that TB is spread only through **AIR** (coughing/sneezing).
        """)