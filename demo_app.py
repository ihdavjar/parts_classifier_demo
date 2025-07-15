import streamlit as st
import time
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from helper_function import (
    create_prompt,
    create_prompt_RAG,
    get_model_output,
    load_db,
    get_model_output_BERT
)
from categories import norm_possible_categories

categories = [temp.lower() for temp in norm_possible_categories]

# --- SETUP ---
st.set_page_config(page_title="Part Classifier", layout="centered")
st.title("Part Category Classifier")

st.markdown("""
Classifies a part into a standard category using its:
- **Description**
- **Manufacturer**
- **Raw Categories (optional)**

Choose between different models (Gemini, GPT, etc.) and optionally enable RAG (Retrieval-Augmented Generation) for context-aware prediction.
""")

# --- SIDEBAR OPTIONS ---
st.sidebar.header("⚙️ Configuration")

api_keys = {
    "gemini-2.5-flash": st.secrets["GOOGLE_API_KEY"],
    "gpt-4.1": st.secrets["OPENAI_API_KEY"],
    "o4-mini": st.secrets["OPENAI_API_KEY"],
    "DeepSeek-V3": st.secrets["DEEPSEEK_API_KEY"],
    "DeepSeek-R1": st.secrets["DEEPSEEK_API_KEY"]
}

model_name = st.sidebar.selectbox(
    "Choose a model",
    options=["gemini-2.5-flash", "gpt-4.1", "o4-mini", "DeepSeek-V3", "DeepSeek-R1", "BERT"]
)

is_bert = model_name == "BERT"
# Disable RAG toggle if BERT is selected
rag_toggle_state = False if is_bert else st.sidebar.toggle("🔄 Enable RAG (for prompt models only)", value=False)
top_k = st.sidebar.slider("Top K retrieved examples (RAG)", 1, 10, 5) if not is_bert else None

# --- INPUT FORM ---
with st.form("part_input_form"):
    manufacturer = st.text_input("Manufacturer", placeholder="e.g. TE CONNECTIVITY")
    description = st.text_area("Description", placeholder="e.g. Automotive Connector Locks...")
    raw_category = st.text_area("Raw Categories", placeholder="e.g. Connectors, Cable Clamps")

    submitted = st.form_submit_button("🔍 Classify")

# --- RUN ---
if submitted:
    if not manufacturer or not description:
        st.error("❗ Please provide both Manufacturer and Description.")
    else:
        with st.spinner("Running inference..."):
            start = time.time()

            try:
                if is_bert:
                    # Call local BERT
                    label, confidence = get_model_output_BERT(
                        manufacturer=manufacturer,
                        description=description,
                        raw_category=raw_category or "None"
                    )
                    duration = time.time() - start
                    st.success("Classification with BERT")
                    st.markdown(f"**Predicted Label:** `{label}`")
                    st.markdown(f"**Confidence Score:** `{confidence:.4f}`")
                    st.markdown(f"Inference Time: `{duration:.2f}` seconds")

                else:
                    # Prompt-based models (with/without RAG)
                    collection = load_db() if rag_toggle_state else None


                    if rag_toggle_state:
                        prompt = create_prompt_RAG(
                            collection=collection,
                            categories=categories,
                            description=description,
                            manufacturer=manufacturer,
                            raw_category=raw_category or "None",
                            k=top_k
                        )
                    else:
                        prompt = create_prompt(
                            manufacturer=manufacturer,
                            description=description,
                            raw_category=raw_category or "None"
                        )

                    prediction = get_model_output(
                        prompt=prompt,
                        RAG=rag_toggle_state,
                        api_key=api_keys[model_name],
                        model_name=model_name
                    )
                    duration = time.time() - start

                    # Output
                    st.success("✅ Classification Completed")
                    st.markdown(f"**Predicted Category:** `{prediction}`")
                    st.markdown(f"Inference Time: `{duration:.2f}` seconds")

                    # Show prompt and optionally RAG sources
                    with st.expander("Show Prompt Used"):
                        st.code(prompt)

                    if rag_toggle_state:
                        st.markdown("📚 Top Retrieved RAG Examples:")
                        data_retrieved, metadata = collection.query(
                            query_texts=[description],
                            n_results=top_k
                        )["documents"][0], collection.query(
                            query_texts=[description],
                            n_results=top_k
                        )["metadatas"][0]

                        for i, (desc, meta) in enumerate(zip(data_retrieved, metadata), 1):
                            st.markdown(f"**Example {i}**")
                            st.markdown(f"- Description: {desc}")
                            st.markdown(f"- Manufacturer: `{meta.get('manufacturer_name', '')}`")
                            st.markdown(f"- Raw Categories: `{meta.get('raw_categories', '')}`")
                            st.markdown(f"- Label: `{meta.get('category', '')}`")

            except Exception as e:
                st.error(f"❌ Error occurred: {str(e)}")