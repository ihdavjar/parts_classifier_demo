import streamlit as st
import google.generativeai as genai
import io
from helper_function import create_prompt, initialize_model, get_model_output, load_config

# --- Load Config & API ---
config = load_config()
api_key = st.secrets["GOOGLE_API_KEY"]
model_name = config.get("model", "gemini-2.5-flash")
model = initialize_model(api_key)

# --- Page Config ---
st.set_page_config(page_title="Part Classifier", layout="centered")
st.title("🔧 AI Part Category Classifier")

# --- Intro ---
st.markdown("""
Welcome to the **AI-Powered Part Classification Tool**.

This tool classifies electronic parts into one of the predefined categories using:
- 🏷️ **Part Description**
- 🏭 **Manufacturer**
- 🗂️ (Optional) Raw Categories

> 🔍 Powered by **Google's Gemini AI** with few-shot learning.
""")

# --- System Overview ---
with st.expander("📊 What Happens Behind the Scenes?"):
    st.markdown("""
1. **User Inputs** are collected: Manufacturer, Description, and optionally Raw Categories.
2. 🧠 A prompt is carefully constructed to explain the task to the AI.
3. 🔄 This prompt is sent to **Gemini**, Google's large language model.
4. 🎯 Gemini responds with the **most suitable category** for the given part.
5. ✅ The app displays the result and lets you download a classification report.

> This makes it easy for engineers, managers, or procurement teams to classify parts instantly!
""")

# --- Input Form ---
with st.form("part_input_form"):
    manufacturer = st.text_input("🏭 Manufacturer", placeholder="e.g. AMPHENOL")
    description = st.text_area("📝 Description", placeholder="e.g. 9 Position D-Sub Receptacle, Female Sockets Connector...")
    raw_category = st.text_area("🗂️ Raw Categories (Optional)", placeholder="e.g. Connectors, Interconnects...")

    submitted = st.form_submit_button("🚀 Classify Part")

# --- Process Submission ---
if submitted:
    if not description or not manufacturer:
        st.error("⚠️ Please fill in both Manufacturer and Description.")
    else:
        prompt = create_prompt(manufacturer, description, raw_category or "None")

        try:
            prediction = get_model_output(model, prompt)

            # --- Show Result ---
            st.success("✅ Classification Completed")
            st.markdown("### 🏷️ Predicted Category:")
            st.markdown(f"""
            <div style='padding: 10px; border: 2px solid #4CAF50; border-radius: 10px;
            background-color: #e8f5e9; text-align: center;'>
                <span style='font-size: 24px; color: #2e7d32;'>{prediction}</span>
            </div>
            """, unsafe_allow_html=True)

            # --- Optional Prompt Viewer ---
            with st.expander("📄 View AI Prompt"):
                st.code(prompt, language="markdown")

            # --- Report Download ---
            report = f"""
Part Classification Report
--------------------------
Manufacturer  : {manufacturer}
Description   : {description}
Raw Categories: {raw_category or 'None'}

✅ Predicted Category: {prediction}
Model Used   : {model_name}
"""
            st.download_button("📥 Download Classification Report", io.BytesIO(report.encode()), "classification_report.txt", "text/plain")

            # --- Model Info ---
            st.info(f"🤖 Classified using **Gemini model**: `{model_name}`")

        except Exception as e:
            st.error(f"❌ Error while classifying: {e}")
