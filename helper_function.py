import pickle
from openai import OpenAI
import google.generativeai as genai
from transformers import BertTokenizer, BertForSequenceClassification
from torch import nn
import torch
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()

class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

def create_prompt(manufacturer, description, raw_category):
    """
    Generate a few-shot prompt for classifying a single part using description, manufacturer, source, and raw category.
    """

    prompt = f"""
You are an AI model that classifies each part into exactly one of the Allowed Categories,
using information such as description, manufacturer, and raw categories.

**Allowed Categories**:
['automated crimp tool', 'backshell', 'cable', 'cavity plug', 'connector', 'connector contact', 'connector contact insertion/removal tool', 'crimp tool', 'crimp tool positioner', 'shielded cable', 'shrink tube', 'wire', 'wire seal']


Here are some labeled examples:

**Example 1**
- Description: 1.5 Ton Battery-Powered Open Frame Crimp Tool for Use with Y Dies, Crimp copper and aluminum cable 26 to 10 AWG with DMCÂ® "Y" Dies (M22520/5-XX and others), Head rotates 350Â°
  Manufacturer: DANIELS MANUFACTURING (DMC)
  Raw Categories: Automated crimp tool
Answer:
automated crimp tool

**Example 2**
- Description: Circular MIL Spec Backshells Multi-Con-X Extra Large Backshell .400-.430 OD Cable
  Manufacturer: SWITCHCRAFT
  Raw Categories: Connectors, Interconnects > Circular Connectors > Backshells and Cable Clamps > Circular MIL Spec Backshells
Answer:
backshell

**Example 3**
- Description: 20 AWG, 4C, Unshielded Cable, Communication and Control, Multiconductor, Unshielded, PVC insulation, -20Â°C to 80Â°C, 300 V
  Manufacturer: ALPHA WIRE
  Raw Categories: Not specified
Answer:
cable

**Example 4**
- Description: CAVITY PLUG, WHITE
  Manufacturer: YAZAKI
  Raw Categories: Not specified
Answer:
cavity plug

**Example 5**
- Description: Minitek 89947 series, Receptacle, Black, 50 Cavity
  Manufacturer: AMPHENOL
  Raw Categories: Connector
Answer:
connector

**Example 6**
- Description: Pin Contact 6 AWG Size 4 Crimp Silver::CONTACT PIN 6AWG CRIMP SILVER
  Manufacturer: AMPHENOL
  Raw Categories: Connectors, Interconnects > Circular Connectors > Circular Connector Contacts
Answer:
connector contact

**Example 7**
- Description: Extraction, Removal & Insertion Tools INSERTION/EXTRACTION
  Manufacturer: DANIELS MANUFACTURING (DMC)
  Raw Categories: Connector contact insertion/removal tool
Answer:
connector contact insertion/removal tool

**Example 8**
- Description: MinitekÂ® 2.00mm (26,28,30 AWG) CRIMP HAND TOOL
  Manufacturer: AMPHENOL
  Raw Categories: Crimp tool
Answer:
crimp tool

**Example 9**
- Description: Crimpers / Crimping Tools DIE FOR EBC320 70MM CONT
  Manufacturer: ANDERSON POWER PRODUCTS
  Raw Categories: Crimp tool positioner
Answer:
crimp tool positioner

**Example 10**
- Description: 80 (40 Pair Twisted) Conductor Multi-Pair Cable Gray 28 AWG Foil, Braid 1000.0' (304.8m)::MULTI-PAIR 80CON 28AWG GRY
  Manufacturer: 3M
  Raw Categories: Not specified
Answer:
shielded cable

---

Now classify the following part:

Description: {description}
Manufacturer: {manufacturer}
Raw Categories: {raw_category}
Answer:

**Instructions**:
- Assign exactly one category label from the allowed categories to the part.
- Return only the category name as a plain string.
- Do not return a list, formatting, explanation, or any additional text.
- Just output the single predicted category name.

**Expected Output Format**:
CategoryName

Answer:"""

    return prompt

def get_prompt_few_shotB(categories, example_data, example_labels, description, manufacturer, raw_categorie):
    assert len(example_data) == len(example_labels), "Few-shot examples and labels must be equal in length."

    category_block = f"**Allowed Categories**:\n[{categories}]\n"

    # Build few-shot examples
    few_shot_blocks = []
    for i, ((desc, manu, raw), label) in enumerate(zip(example_data, example_labels), start=1):
        raw_joined = " | ".join(raw) if isinstance(raw, list) else str(raw)
        few_shot_blocks.append(f"""**Example {i}**
- Description: {desc}
  Manufacturer: {manu}
  Raw Categories: {raw_joined}
Answer:
{label}

""")

    # Final prompt
    prompt = f"""You are an AI model that classifies each part into exactly one of the Allowed Categories,
using information such as description, manufacturer, and raw categories.

{category_block}

Here are some labeled examples:

{''.join(few_shot_blocks)}

**Now classify the following part:**

- Description: {description}
  Manufacturer: {manufacturer}
  Raw Categories: {raw_categorie}

**Instructions**:
- Assign exactly one category label from the allowed categories to the part.
- Return only the category name as a plain string.
- Do not return a list, formatting, explanation, or any additional text.
- Just output the single predicted category name.

**Expected Output Format**:
CategoryName

Answer:"""

    return prompt

def load_db():
  chroma_client = chromadb.PersistentClient(path="./chroma_db")

  collection = chroma_client.get_collection(name="rag_collection", embedding_function=SentenceTransformerEmbeddingFunction())

  return collection

def retrieve_relevant_documents(query, collection, k=5):
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return documents, metadatas


def create_prompt_RAG(collection, categories, description, manufacturer, raw_category, k = 5):

    data_retrieved, metadata = retrieve_relevant_documents(description, collection, k)
    example_manufacturers = [meta.get("manufacturer_name", "") for meta in metadata]
    example_descriptions = data_retrieved
    example_raw_categories = [meta.get("raw_categories", "").split(" | ") for meta in metadata]
    example_labels = [meta.get("category", "") for meta in metadata]
    example_data = list(zip(example_descriptions, example_manufacturers, example_raw_categories))

    prompt = get_prompt_few_shotB(categories, example_data, example_labels, description, manufacturer, raw_category)

    return prompt


def get_model_output(prompt, RAG = False, api_key = None, model_name="gemini-2.5-flash"):

    if not api_key:
        raise ValueError("API key is required to initialize the model.")

    if model_name == "gemini-2.5-flash":
        genai.configure(api_key = api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        predictions = response.candidates[0].content.parts[0].text.strip().lower()


    elif model_name == "gpt-4.1":
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}],
        )

        predictions = response.choices[0].message.content.strip()
        predictions = predictions.split("\n")[0].strip().lower()  # Get the first line as the prediction

    elif model_name == "o4-mini":
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="o4-mini",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}],
        )

        predictions = response.choices[0].message.content.strip()
        predictions = predictions.split("\n")[0].strip().lower()  # Get the first line as the prediction

    elif model_name == "DeepSeek-V3":
        client = OpenAI(
                  base_url="https://api.studio.nebius.com/v1/",
                  api_key=api_key
              )

        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}],
        )

        raw_output = response.choices[0].message.content.strip()

        # Extract only the actual prediction part (after </think>)
        if "</think>" in raw_output:
            raw_output = raw_output.split("</think>")[-1]

        predictions = raw_output.strip().lower()

    elif model_name == "DeepSeek-R1":
        client = OpenAI(
                  base_url="https://api.studio.nebius.com/v1/",
                  api_key=api_key
              )
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-0528",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}],
        )

        raw_output = response.choices[0].message.content.strip()

        # Extract only the actual prediction part (after </think>)
        if "</think>" in raw_output:
            raw_output = raw_output.split("</think>")[-1]

        predictions = raw_output.strip().lower()

    return predictions


## BERT Classifier
def predict_label(text, model, tokenizer, label_encoder):

    model.eval()

    # Tokenize input
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # Get predicted class index and confidence score
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence_score = probs[0][predicted_class].item()

    label = label_encoder.inverse_transform([predicted_class])[0]
    return label, confidence_score

def get_model_output_BERT(manufacturer, description, raw_category):
    """
    Get model output using BERT for sequence classification.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERTClassifier(num_classes=13)
    model.load_state_dict(torch.load('best_model_13_100.pt', map_location=torch.device('cpu')))

    if manufacturer == "":
        manufacturer = "Not provided"
    if description == "":
        description = "Not provided"
    if raw_category == "":
        raw_category = "Not provided"

    input_text = f"Manufacturer: {manufacturer}. Description: {description}. Raw Categories: {raw_category}"

    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)

    label, confidence_score = predict_label(input_text, model, tokenizer, label_encoder)

    return label, confidence_score
