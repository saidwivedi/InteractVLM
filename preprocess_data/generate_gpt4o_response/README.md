# HOI-VQA Generation using GPT-4o

This folder contains scripts and data for generating Human-Object Interaction Visual Question Answering (HOI-VQA) datasets using OpenAI's GPT-4o model. The generated HOI-VQA data is used to train InteractVLM models as mentioned in the main README.

## 📁 Contents

### Scripts

1. **`get_gpt4o_prompts.py.py`** - Main script for generating HOI-VQA responses using Azure OpenAI GPT-4o API
2. **`convert_gpt4o_to_vqa.py`** - Converts raw GPT-4o responses to standard VQA format

### Data Files

1. **`damon_gpt4o.txt`** - Raw GPT-4o responses for DAMON dataset
2. **`lemon_gpt4o.txt`** - Raw GPT-4o responses for LEMON dataset  
3. **`piad_gpt4o.txt`** - Raw GPT-4o responses for PIAD dataset

## 🔧 Setup

### Prerequisites

1. **Azure OpenAI Access**: You need access to Azure OpenAI API with GPT-4o deployment
2. **API Configuration**: Update the following variables in `get_gpt4_prompts.py.py`:
   ```python
   API_BASE = 'YOUR_AZURE_OPENAI_API_BASE_URL'
   API_KEY = "YOUR_AZURE_OPENAI_API_KEY"
   DEPLOYMENT_NAME = 'YOUR_DEPLOYMENT_NAME'
   API_VERSION = '2023-05-15'
   ```

### Dependencies

```bash
pip install openai pillow numpy
```

## 🚀 Usage

### Step 1: Generate HOI-VQA Responses

Use `get_gpt4_prompts.py.py` to generate responses for different datasets:

```bash
# For DAMON dataset
python get_gpt4_prompts.py.py  # Uncomment generate_for_damon() in main

# For LEMON dataset  
python get_gpt4_prompts.py.py  # Uncomment generate_for_lemon() in main

# For PIAD dataset
python get_gpt4_prompts.py.py  # Uncomment generate_for_piad_seen() in main
```

### Step 2: Convert to VQA Format

Convert the raw GPT-4o responses to standard VQA conversation format:

```bash
python convert_gpt4o_to_vqa.py --input damon_gpt4o.txt --output damon_vqa.json
python convert_gpt4o_to_vqa.py --input lemon_gpt4o.txt --output lemon_vqa.json
python convert_gpt4o_to_vqa.py --input piad_gpt4o.txt --output piad_vqa.json
```

## 📋 VQA Questions

The script generates responses for 5 structured questions per image-object pair:

1. **HVisual**: "Describe the human in terms of clothing, appearance or any distinctive feature."
2. **HContact**: "What part of the human's body is in contact with the {object}?"
3. **Interaction**: "Describe the interaction of human with {object}?"
4. **OVisual**: "Can you describe the {object} in terms of shape, color or distinctive feature?"
5. **OContact**: "Which part of the {object} is in contact with human?"

## 📊 Data Format

### Raw Response Format (`*.txt` files)
```
image_name.jpg,object_name-HVisual: description\nHContact: description\nInteraction: description\nOVisual: description\nOContact: description
```

### VQA Conversation Format (`*.json` files)
```json
{
  "id": "image_name",
  "image": "path/to/image.jpg",
  "conversations": [
    {
      "from": "human", 
      "value": "<image>\nDescribe the human in terms of clothing, appearance or any distinctive feature."
    },
    {
      "from": "gpt",
      "value": "The human is wearing a white T-shirt and cargo shorts."
    },
    ...
  ]
}
```
