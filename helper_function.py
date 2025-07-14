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
Adhesive, Cable tie, Connector, Connector kit, Connector accessories, Backshell, Backshell adapter, Backshell strain relief clamp, Cable support sleeve, Cavity plug, Connector contact, Connector insert, Connector jam nut, Connector Band Clamp, Connector dummy stowage, Connector Grommet, Connector Wedge Lock, Contact member, End cap, Gasket, Hardware, Hose clamp, Keying Pin, Mounting clip, Boot, Safety wire, Wire seal, Shield split support ring, Shield band, Shrink boot, Spacer, Split Ring, Strain relief accessory, Tinel Lock Ring, Device, Device with 2 axial leads, Fixing device, Label, Part selection, Part selection by filter, Part selection from set, Segment/bundle materials, Cable, Flat ribbon cable, Shielded cable, Coaxial cable, Differential pair cable, Triaxial cable, Cable gland, Cable/wire connecction, Insulation piercing connector, Shield connector (splice), Solder sleeve device, Splice (Inline), Ferrite, Filler, Fishnet, Grommet, Lacing tape, Protective covering, Convoluted tube, Overbraid, Shrink tube, Shrink tube marker sleeve, Sleeving, Expando sleeve, Fire protection sleeve, Shrink transition, Wire, Tape, Terminal lug, Contact terminal, Ferrule Terminal, Ring Terminal, Spade Terminal, Tool, Automated crimp tool, Banding tool, Connector contact insertion/removal tool, Connector contact insertion tool, connector contact removal tool, Contact retention test tool, Crimp tool, Crimp tool positioner, Latch disengaging tool, Printer, Torque adapter, Unknown


Here are some labeled examples:

**Example 1**
- Description: ::Maximum Bond Krazy Glue, 0.18 oz
  Manufacturer: KRAZY GLUE
  Raw Categories: Glue, Adhesives, Applicators | Tapes, Adhesives, Materials
Answer:
Adhesive

**Example 2**
- Description: 9 Position D-Sub Receptacle, Female Sockets Connector::D-SUB SMT 9PIN STRAIGHT FEMALE,
  Manufacturer: HARTING
  Raw Categories: Connectors, Interconnects | D-Sub Connector Assemblies | D-Sub Connector Assemblies | D-Sub Connector Assemblies | D-Sub, D-Shaped Connectors
Answer:
Connector

**Example 3**
- Description: Connector Backshell, Adapter 1 1/5-18 UNEF 24::ADAPTER
  Manufacturer: TE CONNECTIVITY
  Raw Categories: Backshells and Cable Clamps | Backshells and Cable Clamps | Circular Connectors | Connectors, Interconnects
Answer:
Backshell

**Example 4**
- Description: Adapter Coaxial Connector BNC Plug, Male Pin To BNC Plug, Male Pin 50 Ohms::BNC (M-M)
  Manufacturer: POMONA ELECTRONICS
  Raw Categories: Coaxial Connector (RF) Adapters | Coaxial Connector (RF) Adapters | Coaxial Connectors (RF) | Connectors, Interconnects
Answer:
Backshell adapter

**Example 5**
- Description: Sealing Plug, PBT GF30, White, Size 12/16, -55 – 125 °C [-67 – 257 °F] Operating Temperature, DEUTSCH
  Manufacturer: TE CONNECTIVITY
  Raw Categories: Cavity plug
Answer:
Cavity plug

**Example 6**
- Description: Pin, Outer Contact Contact Coax Tin Crimp::CONN PIN OUTER CONTACT CRIMP TIN
  Manufacturer: JAE ELECTRONICS
  Raw Categories: Coaxial Connector (RF) Contacts | Coaxial Connector (RF) Contacts | Coaxial Connector (RF) Contacts | Coaxial Connectors (RF) | Coaxial Connectors (RF) | Connectors, Interconnects | Connectors, Interconnects
Answer:
Connector contact

**Example 7**
- Description: HE-006-FSS
  Manufacturer: TE CONNECTIVITY
  Raw Categories: Connectors | Power Connectors | Rectangular Contact Inserts | Rectangular Power
Answer:
Connector insert

**Example 8**
- Description: Automotive Connector Locks & Position Assurance, Secondary Lock, Orange, PBT, 3 Position, -55 – 125 °C [-67 – 257 °F], DEUTSCH DTM
  Manufacturer: TE CONNECTIVITY
  Raw Categories: Connector Wedge Lock
Answer:
Connector Wedge Lock

**Example 9**
- Description: Connector Cap (Cover), Shell For HD Series::HD1 SERIES LVDS PLUG COVER SHELL
  Manufacturer: JAE ELECTRONICS
  Raw Categories: Connectors, Interconnects | Connectors, Interconnects | Rectangular Connector Accessories | Rectangular Connector Accessories | Rectangular Connector Accessories | Rectangular Connector Accessories | Rectangular Connector Accessories | Rectangular Connectors | Rectangular Connectors
Answer:
End cap

**Example 10**
- Description: Gasket, Closed Cell Sponge, Black, 12 Position Receptacle, -57 – 107 °C [-70 – 225 °F] Operating Temperature, DEUTSCH DT/DTM
  Manufacturer: TE CONNECTIVITY
  Raw Categories: Gasket
Answer:
Gasket

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
    model = BERTClassifier(num_classes=88)
    model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))

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
