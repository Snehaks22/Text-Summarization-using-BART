from flask import Flask, render_template, request, jsonify
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

app = Flask(__name__)

# Load the pre-trained BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    if not text.strip():
        return jsonify({'summary': 'Error: No text provided for summarization.'})

    # Tokenize input text and get its length
    input_tokens = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    input_length = input_tokens.shape[1]  # Number of tokens in input

    # Set summary length as 1/3rd of input length
    summary_length = max(50, input_length // 3)  # Ensuring a minimum length of 50 tokens

    # Generate summary
    summary_ids = model.generate(
        input_tokens, 
        max_length=summary_length,  # Dynamic summary length
        min_length=summary_length - 20,  # Allow slight variation in size
        length_penalty=1.5, 
        num_beams=6,  
        no_repeat_ngram_size=3,  
        early_stopping=True  
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
