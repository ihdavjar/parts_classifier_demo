import streamlit as st
import google.generativeai as genai
from helper_function import create_prompt, initialize_model, get_model_output, load_config


# Load API key from config
config = load_config()
api_key = config.get("api_key")
model_name = config.get("model", "gemini-2.5-flash")

# Initialize the model
model = initialize_model(api_key)

st.set_page_config(page_title="Part Classifier", layout="centered")
st.title("AI Part Category Classifier")

st.markdown("""
This tool classifies electronic parts into one of the standard categories using:
- Part Description
- Manufacturer
- Optional Raw Categories

**All processing is done by Google's Gemini model with few-shot examples.**
""")

# --- Input Section ---
with st.form("part_input_form"):
    manufacturer = st.text_input("Manufacturer", placeholder="e.g. AMPHENOL")
    description = st.text_area("Description", placeholder="e.g. 9 Position D-Sub Receptacle, Female Sockets Connector...")
    raw_category = st.text_area("Raw Categories (Optional)", placeholder="Optional raw categories, e.g. Connectors, Interconnects...")

    submitted = st.form_submit_button("Classify Part")

# --- Run Classification ---
if submitted:
    if not description or not manufacturer:
        st.error("Please fill in Manufacturer, Source, and Description.")
    else:
        # Format prompt
        prompt = create_prompt(manufacturer, description, raw_category or "None")

        # Call Gemini model
        try:
            prediction = get_model_output(model, prompt)
            st.success("✅ Classification Result:")
            st.markdown(f"**Predicted Category:** `{prediction}`")
        except Exception as e:
            st.error(f"❌ Error while classifying: {e}")
