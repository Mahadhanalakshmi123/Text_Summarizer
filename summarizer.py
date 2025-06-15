from transformers import BartTokenizer, BartForConditionalGeneration
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_text_input(text):
    return generate_summary(text)

def summarize_pdf_file(file_obj):
    doc = fitz.open(stream=file_obj.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    return generate_summary(text)

def summarize_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join([p.get_text() for p in soup.find_all("p")])
        return generate_summary(text)
    except:
        return "Failed to extract text from URL."