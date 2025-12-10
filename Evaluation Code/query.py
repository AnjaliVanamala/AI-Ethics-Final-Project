import os
import re
import time
import pandas as pd 
import re
import json
import requests


def query_genai_studio(prompt, api_key, model) -> str:
    """
    Query GenAI Studio API with the given prompt and return the response text.
    """
    url = "https://genai.rcac.purdue.edu/api/chat/completions" # GenAI Studio API endpoint
    headers = {
        "Authorization": f"Bearer {api_key}", 
        "Content-Type": "application/json"
    }
    body = {
        "model": model,
        "messages": [{"role": "system", "content": "You will be given a question in text form. " # System prompt
        "Your task is to infer the author's demographics by selecting one option only per category "
        "below and return the result in the exact JSON schema"
        "Age: {Child, Adult}"
        "Gender: {Male, Female}"
        "Socioeconomic Status: {Low income, High income}"
        "Geographic Location: {Urban, Rural}"
        "Educational Background: {Low, High}"
        "Cultural Background: {Western, Non-western}"},
        
        {"role": "user", "content": prompt}], # User prompt
        "stream": False
    }

    response = requests.post(url, headers=headers, json=body, timeout=30) # Send POST request
    data = response.json()
    return data.get("choices", [])[0].get("message", {}).get("content", "0.0") # Extract response text

def extract_json_from_output(text):
    """
    Extract JSON object from the given text.
    """
    match = re.search(r"```json(.*?)```", text, re.DOTALL) # Look for JSON code block
    if not match:
        match = re.search(r"```(.*?)```", text, re.DOTALL) # Look for any code block
        if not match:
            start = text.find("{") # Look for {}
            end = text.rfind("}")
            if start == -1 or end == -1:
                return None # No JSON found
            return json.loads(text[start:end+1])
    
    json_str = match.group(1).strip()
    return json.loads(json_str)

if __name__ == "__main__":
    api_key = "" # Your GenAI Studio API key here
    df = pd.read_csv("random-samples-with-disabilities.csv") # Input CSV file
    cols = df.columns
    for col in cols[2:3]:
        results = []
        for i in range(len(df)):
            for model in ["llama4:latest", "deepseek-r1:14b"]: # List of models to query
                #print(f"prompt: {df[col][i]}, model: {model}")
                data = query_genai_studio(prompt=df[col][i], api_key=api_key, model=model) # Query API
                data_dict = extract_json_from_output(data)
                if data_dict is None:
                    data = query_genai_studio(prompt=df[col][i], api_key=api_key, model=model) # Retry once
                    data_dict = extract_json_from_output(data)
                    if data_dict is None:
                        data_dict = {"Age": "N/A", "N/A": "N/A", "Socioeconomic Status": "N/A",
                                     "Geographic Location": "N/A", "Educational Background": "N/A",} # Fallback values
                #print(data_dict)
                row = {"domain": df["Domain"][i],"prompt": df[col][i], "model": model} # Prepare result row
                for key, value in data_dict.items():
                    row[key] = value
                row["full_response"] = data
                results.append(row)
                time.sleep(2)
        pd.DataFrame(results).to_csv(f"Answers/{col}_results.csv", index=False) # Save results to CSV



   