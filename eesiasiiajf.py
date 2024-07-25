import os
import json
import re
import time
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import google.generativeai as genai
import backoff

# BERT setup
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
bert_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
nlp = pipeline("ner", model=bert_model, tokenizer=tokenizer)

# Gemini setup
GOOGLE_API_KEY = 'AIzaSyADjiW040x08E3Pox6DgsPl9E_88EZ_0d4'
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

def extract_signature(email_text: str) -> Dict[str, Dict[str, str]]:
    """
    Extract signature information using a hybrid approach of BERT and Gemini.
    """
    bert_result = extract_signature_bert(email_text)
    gemini_result = extract_signature_gemini(email_text)
    
    # Combine results, preferring Gemini's output when available
    combined_result = {**bert_result, **gemini_result}
    
    # Post-process the combined result
    post_processed_result = post_process_signature(combined_result, email_text)
    
    return {
        "bert": bert_result,
        "gemini": gemini_result,
        "hybrid": post_processed_result
    }

def extract_signature_bert(email_text: str) -> Dict[str, str]:
    """
    Extract signature information using BERT NER model and regex.
    """
    lines = email_text.strip().split('\n')
    signature_start = -1
    for i, line in enumerate(lines):
        if re.search(r'\b(regards|sincerely|cheers|best|thanks)\b', line.lower()):
            signature_start = i + 1
            break
    
    if signature_start == -1:
        return {}
    
    signature_text = '\n'.join(lines[signature_start:])
    ner_results = nlp(signature_text)
    
    signature = {}
    current_name = []
    current_org = []

    for item in ner_results:
        if item['entity'].endswith('-PER'):
            current_name.append(item['word'])
        elif item['entity'].endswith('-ORG'):
            current_org.append(item['word'])
        else:
            if current_name:
                signature['name'] = ' '.join(current_name)
                current_name = []
            if current_org:
                signature['organization'] = ' '.join(current_org)
                current_org = []

    if current_name:
        signature['name'] = ' '.join(current_name)
    if current_org:
        signature['organization'] = ' '.join(current_org)

    return signature

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def extract_signature_gemini(email_text: str) -> Dict[str, str]:
    """
    Extract signature information using Gemini model with error handling and retries.
    """
    prompt = f"""
    Extract the signature information from the following email and format it as JSON.
    Include fields for name, job title, organization, email, and phone number if available.
    If a field is not present, omit it from the JSON.
    If there is no signature information, return an empty JSON object.

    Email:
    {email_text}

    JSON format:
    {{
        "name": "...",
        "job_title": "...",
        "organization": "...",
        "email": "...",
        "phone": "..."
    }}
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        json_str = response.text.strip()
        if json_str.startswith('```json'):
            json_str = json_str[7:-3]  # Remove ```json and ``` 
        signature_json = json.loads(json_str)
        return signature_json
    except Exception as e:
        print(f"Error in Gemini extraction: {str(e)}")
        return {}

def post_process_signature(signature: Dict[str, str], email_text: str) -> Dict[str, str]:
    """
    Post-process the extracted signature to improve accuracy and completeness.
    """
    # Clean up organization name
    if 'organization' in signature:
        signature['organization'] = re.sub(r'##\w+\s*', '', signature['organization']).strip()
    
    # Extract email if not already present
    if 'email' not in signature:
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', email_text)
        if email_match:
            signature['email'] = email_match.group()
    
    # Extract phone if not already present
    if 'phone' not in signature:
        phone_match = re.search(r'\+?1?\s*\(?-*\.*\s*\d{3}\s*\(?-*\.*\s*\d{3}\s*-*\.*\s*\d{4}', email_text)
        if phone_match:
            signature['phone'] = phone_match.group()
    
    # Extract job title if not already present
    if 'job_title' not in signature:
        job_title_match = re.search(r'\n([\w\s]+)\n', email_text)
        if job_title_match:
            signature['job_title'] = job_title_match.group(1).strip()
    
    return signature

def run_tests(email_texts: List[str], expected_results: List[Dict[str, str]]):
    """
    Run the test cases and evaluate the hybrid model's responses.
    """
    total_score = {"bert": 0, "gemini": 0, "hybrid": 0}
    max_score_per_test = 10

    for i, email in enumerate(email_texts):
        print(f"\nEmail {i + 1}:")
        print("=" * 50)
        print(email)
        print("\nExpected Result:")
        print(json.dumps(expected_results[i], indent=2))

        print("\nBERT Extraction Result:")
        bert_result = extract_signature_bert(email)
        print(json.dumps(bert_result, indent=2))
        
        print("\nGemini Extraction Result:")
        gemini_result = extract_signature_gemini(email)
        print(json.dumps(gemini_result, indent=2))
        
        print("\nHybrid Extraction Result:")
        hybrid_result = extract_signature(email)
        print(json.dumps(hybrid_result["hybrid"], indent=2))
        
        bert_score = grade_extraction(bert_result, expected_results[i])
        gemini_score = grade_extraction(gemini_result, expected_results[i])
        hybrid_score = grade_extraction(hybrid_result["hybrid"], expected_results[i])

        print(f"BERT Score: {bert_score}/{max_score_per_test}")
        print(f"Gemini Score: {gemini_score}/{max_score_per_test}")
        print(f"Hybrid Score: {hybrid_score}/{max_score_per_test}")
        
        total_score["bert"] += bert_score
        total_score["gemini"] += gemini_score
        total_score["hybrid"] += hybrid_score

    print("\nOverall Scores:")
    print(f"BERT: {total_score['bert']} / {len(email_texts) * max_score_per_test}")
    print(f"Gemini: {total_score['gemini']} / {len(email_texts) * max_score_per_test}")
    print(f"Hybrid: {total_score['hybrid']} / {len(email_texts) * max_score_per_test}")

def grade_extraction(model_result: Dict[str, str], expected_result: Dict[str, str]) -> int:
    """
    Grade the model's extraction result against the expected result.
    Returns a score out of 10.
    """
    score = 0
    total_fields = 5  # name, job title, organization, email, phone

    for field in ["name", "job_title", "organization", "email", "phone"]:
        if field in expected_result:
            if field in model_result and model_result[field] == expected_result[field]:
                score += 2  # Full points for correct extraction
            elif field in model_result:
                score += 1  # Partial points for incorrect extraction
        else:
            if field not in model_result:
                score += 2  # Full points for correctly omitting the field

    return score

# Test emails and expected results (same as before)
test_emails = [
    """
    Dear John,

    I hope this email finds you well. Here's the report you requested.

    Best regards,
    Jane Smith
    Senior Data Analyst
    TechCorp Inc.
    jane.smith@techcorp.com
    +1 (555) 123-4567
    """,
    """
    Hi team,

    Great job on the project! Let's discuss next steps in our meeting tomorrow.

    Cheers,
    Alex
    """,
    """
    Hi Mike,

    Could you please send me the latest updates on the project?

    Thanks,
    Sarah
    """,
    """
    Hello,

    Please find attached the latest financial report for Q1.

    Regards,
    John Doe
    Chief Financial Officer
    FinanceCorp
    john.doe@financecorp.com
    +44 20 7946 0958
    """,
    """
    Hey,

    Just checking in to see if you're available for a quick call later.

    Thanks,
    Emily Davis
    Marketing Specialist
    MarketGurus Ltd.
    emily.davis@marketgurus.com
    """
]

expected_results = [
    {
        "name": "Jane Smith",
        "job_title": "Senior Data Analyst",
        "organization": "TechCorp Inc.",
        "email": "jane.smith@techcorp.com",
        "phone": "+1 (555) 123-4567"
    },
    {
        "name": "Alex"
    },
    {
        "name": "Sarah"
    },
    {
        "name": "John Doe",
        "job_title": "Chief Financial Officer",
        "organization": "FinanceCorp",
        "email": "john.doe@financecorp.com",
        "phone": "+44 20 7946 0958"
    },
    {
        "name": "Emily Davis",
        "job_title": "Marketing Specialist",
        "organization": "MarketGurus Ltd.",
        "email": "emily.davis@marketgurus.com"
    }
]

# Run tests
run_tests(test_emails, expected_results)
