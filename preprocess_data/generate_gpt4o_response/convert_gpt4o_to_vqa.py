import json
import os
import argparse

def parse_input_line(line):
    # Split the line into image_path and content
    try:
        image_path, content = line.strip().split(',', 1)
        
        # Extract the class name and the rest of the content
        class_name, qa_content = content.split('-', 1)
        
        # Split the content into different sections
        sections = qa_content.split('\\n')
        
        # Create a dictionary to store the QA pairs
        qa_dict = {}
        for section in sections:
            if ':' in section:
                key, value = section.split(':', 1)
                qa_dict[key.strip()] = value.strip()
        
        # Check if all required fields are present and non-empty
        required_fields = ['HVisual', 'HContact', 'Interaction', 'OVisual', 'OContact']
        for field in required_fields:
            if field not in qa_dict or not qa_dict[field].strip():
                return None
        
        base_name = os.path.basename(image_path)[:-4]
        
        return {
            "id": base_name,
            "image": image_path,
            "class_name": class_name,
            "qa_pairs": qa_dict
        }
    except:
        return None

def create_conversation_format(parsed_data):
    conversations = []
    
    # Define the questions in order
    questions = [
        ("HVisual", "Describe the human in terms of clothing, appearance or any distinctive feature."),
        ("HContact", f"What part of the human's body is in contact with the {parsed_data['class_name']}?"),
        ("Interaction", f"Describe the interaction of human with {parsed_data['class_name']}?"),
        ("OVisual", f"Can you describe the {parsed_data['class_name']} in terms of shape, color or distinctive feature?"),
        ("OContact", f"Which part of the {parsed_data['class_name']} is in contact with human?")
    ]
    
    # Create conversation pairs
    for q_key, q_text in questions:
        conversations.append({
            "from": "human",
            "value": f"<image>\n{q_text}" if len(conversations) == 0 else q_text
        })
        conversations.append({
            "from": "gpt",
            "value": parsed_data['qa_pairs'].get(q_key, "")
        })
    
    return {
        "id": parsed_data["id"],
        "image": parsed_data["image"],
        "conversations": conversations
    }

def process_data(input_file, output_file):
    result = []
    skipped = 0
    processed = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                parsed_data = parse_input_line(line)
                if parsed_data is not None:
                    formatted_data = create_conversation_format(parsed_data)
                    result.append(formatted_data)
                    processed += 1
                else:
                    skipped += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete:")
    print(f"- Processed entries: {processed}")
    print(f"- Skipped entries: {skipped}")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GPT-4 output to VQA format")
    parser.add_argument("--input", type=str, help="Path to input file", default="input_data.txt")
    parser.add_argument("--output", type=str, help="Path to output file", default="output_data.json")
    args = parser.parse_args()
    
    process_data(args.input, args.output)