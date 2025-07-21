import base64
from mimetypes import guess_type
from PIL import Image
import time
import os
import numpy as np
import json
from os.path import join
from openai import AzureOpenAI

API_BASE = 'API_BASE' # Replace with your Azure OpenAI API base URL
API_KEY = "API_KEY"  # Replace with your Azure OpenAI API key
DEPLOYMENT_NAME = 'DEPLOYMENT_NAME'  # Replace with your Azure OpenAI deployment name
API_VERSION = '2023-05-15'  # Replace with your Azure OpenAI API version
TOKEN_LIMIT_PER_MIN = 15000


def get_openai_client():
    return AzureOpenAI(
        api_key=API_KEY,  
        api_version=API_VERSION,
        base_url=f"{API_BASE}openai/deployments/{DEPLOYMENT_NAME}",
    )


# Function to encode a local image into a base64 data URL
def local_image_to_data_url(image_path, max_size=(256, 256)):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    # Open, resize, and encode the image
    with Image.open(image_path) as img:
        img.thumbnail(max_size)  # Resize the image to reduce token usage
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def write_response_to_file(write_path, img_name, obj_name, response):
    response = response.replace('\n', '\\n')
    with open(write_path, 'a') as file:
        file.write(f"{img_name},{obj_name}-{response}\n")

# Function to call Azure GPT4o API
def generate_attributes(client, image_url, class_name):
    messages = [
        { "role": "system", "content": "You are a helpful assistant. Answer each question in the format: 'keyword: description'. Keep the format consistent across all answers. The answer should for each question should be one line" },
        { 
            "role": "user", 
            "content": [
                { "type": "text", "text": f"HVisual: Describe the human in terms of clothing, appearance or any distinctive feature." },
                { "type": "text", "text": f"HContact: What part of the human's body is in contact with the {class_name}?" },
                { "type": "text", "text": f"Interaction: Describe the interaction of human with {class_name}?" },
                { "type": "text", "text": f"OVisual: Can you describe the {class_name} in terms of shape, color or distinctive feature?" },
                { "type": "text", "text": f"OContact: Which part of the {class_name} is in contact with human?" },
                { "type": "image_url", "image_url": { "url": image_url } }
            ]
        }
    ]   
    # Call Azure GPT-4o API
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=messages,
        max_tokens=4096
    )
    
    return response.choices[0].message.content, response

# Single call to GPT4o API using single image and object name
def gpt_api_call(client, img_path, img_name, obj_name, write_path, tokens_used_in_min, start_time):
    image_data_url = local_image_to_data_url(img_path)
    response = None
    try:
        attribute, response = generate_attributes(client, image_data_url, obj_name)
        write_response_to_file(write_path, img_name, obj_name, attribute)
        tokens_used_in_min += response.usage.total_tokens
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        write_response_to_file(write_path, img_name, obj_name, "")

    elapsed_time = time.time() - start_time
    if tokens_used_in_min > TOKEN_LIMIT_PER_MIN:
        if elapsed_time < 60:
            print(f"Sleeping for {60 - elapsed_time} seconds")
            time.sleep(60 - elapsed_time)
        tokens_used_in_min = 0
        start_time = time.time()
    return tokens_used_in_min, start_time

# Prepare imagel lists and objnames for LEMON Dataset
def generate_for_lemon():

    DATA_FOLDER='/is/cluster/fast/sdwivedi/work/lemon_3d/Data/lemon/Images/'
    WRITE_PATH = 'lemon_gpt4o.txt'
    img_lists = []
    for obj_fold in sorted(os.listdir(DATA_FOLDER)):
        for afford_fold in sorted(os.listdir(join(DATA_FOLDER, obj_fold))):
            for img_name in sorted(os.listdir(join(DATA_FOLDER, obj_fold, afford_fold))):
                img_path = join(DATA_FOLDER, obj_fold, afford_fold, img_name)
                img_lists.append([obj_fold, img_path])

    print(f"Total number of images in lemon: {len(img_lists)}")
    total_images = len(img_lists)
    return img_lists, total_images, WRITE_PATH

def generate_for_piad_seen():
    DATA_FOLDER='/is/cluster/fast/sdwivedi/work/lemon_3d/Data/piad_ocontact_seen/Img/Train'
    WRITE_PATH = 'piad_gpt4o.txt'
    img_lists = []
    for obj_fold in sorted(os.listdir(DATA_FOLDER)):
        for afford_fold in sorted(os.listdir(join(DATA_FOLDER, obj_fold))):
            for img_name in sorted(os.listdir(join(DATA_FOLDER, obj_fold, afford_fold))):
                img_path = join(DATA_FOLDER, obj_fold, afford_fold, img_name)
                img_lists.append([obj_fold, img_path])

    print(f"Total number of images in lemon: {len(img_lists)}")
    total_images = len(img_lists)
    return img_lists, total_images, WRITE_PATH

# Prepare imagel lists and objnames for DAMON Dataset
def generate_for_damon():
    DATA_FOLDER = '/is/cluster/fast/sdwivedi/work/lemon_3d/Data/damon/train/images/'
    OBJ_CONTACT_ANNO = '/is/cluster/fast/sdwivedi/work/lemon_3d/Data/damon/train/contact_label_objectwise.npy'
    IMGNAME = '/is/cluster/fast/sdwivedi/work/lemon_3d/Data/damon/train/imgname.npy'
    WRITE_PATH = './damon_gpt4o.txt'

    obj_contact_anno = np.load(OBJ_CONTACT_ANNO, allow_pickle=True)
    imgname_anno = np.load(IMGNAME, allow_pickle=True)
    img_lists = []
    count = 0
    special_counter = {}
    for img_idx, img_name in enumerate(imgname_anno):
        img_name = os.path.basename(img_name)
        img_path = join(DATA_FOLDER, img_name)
        count += 1
        num_objects_with_more_than_2_contacts = 0
        for obj_name, contact_vertices in obj_contact_anno[img_idx].items():
            if len(contact_vertices) == 0:
                continue
            if 'supporting' in obj_name:
                continue
            num_objects_with_more_than_2_contacts += 1
            img_lists.append([obj_name, img_path])
        if num_objects_with_more_than_2_contacts >= 2:
            special_counter[img_name] = num_objects_with_more_than_2_contacts

    total_images = len(img_lists)
    print(f"Total number of images in damon: {total_images} {count}")
    print(f'Number of images with more than 2 contacts: {len(list(special_counter.keys()))}')
    return img_lists, total_images, WRITE_PATH


if __name__ == "__main__":

    # Get the list of objects and afford
    # img_lists, total_images, write_path = generate_for_lemon()
    # img_lists, total_images, write_path = generate_for_damon()
    img_lists, total_images, write_path = generate_for_piad_unseen()
    print(write_path)

    client = get_openai_client()
    start_time = time.time()
    tokens_used_in_min = 0

    num_responses_processed = 0
    if os.path.exists(write_path):
        responses_alreay_processed = open(write_path, 'r').readlines()
        num_responses_processed = len(responses_alreay_processed)

    for idx, (obj_name, img_path) in enumerate(img_lists):
        # Encode the image as a data URL
        if idx < num_responses_processed:
            continue
        img_name = img_path.split('/')[-1]
        print(f"Processing {idx}/{total_images}: {obj_name} / {img_name} with token usage: {tokens_used_in_min}")
        tokens_used_in_min, start_time = gpt_api_call(client, img_path, img_name, obj_name, write_path, tokens_used_in_min, start_time)


    
