import requests
import json
import base64
import os
import openai
from dotenv import load_dotenv
load_dotenv()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def GPT4V(people, image_path):

    base64_image = encode_image(image_path)
    openai_api_key = os.getenv("OPENAI_API_KEY")

    url = 'https://api.openai.com/v1/chat/completions'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {openai_api_key}'}

    data = {
        'model': 'gpt-4-vision-preview',
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': 'What’s in this image?'
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        'max_tokens': 300
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    description = response.json()["choices"][0]["message"]["content"]

    openai.api_key=openai_api_key

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are going to be provided a description of what you see, and a list of names of the people you can see. Your job is to explain what you see by creating one, unified description of the image that uses the names of the people in the photo. Keep the description brief, do not make any assumptions. Only use the informatino you are provided. Do not say 'in this photo' or anything similar. Pretend you are describing what you see. Do not say 'I see', just describe it."},
            {"role": "user", "content": f"Description: {description} People/Person: {people}"}
        ]
    )

    return response["choices"][0]["message"]["content"]