import os
import json
import google.generativeai as genai

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
"""
    return prompt

def initialize_model(api_key, model_name="gemini-2.5-flash"):
    """
    Initialize the Generative AI model.
    """
    if not api_key:
        raise ValueError("API key is required to initialize the model.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    return model


def get_model_output(model, prompt):
    """
    Get the model output for the given prompt.
    """

    response = model.generate_content(prompt).text

    return response

def load_config():
    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as f:
        config = json.load(f)
    return config

