import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from googletrans import Translator
from gtts import gTTS
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import pytesseract
import cv2
#nltk.download('stopwords')
#nltk.download('punkt')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

def summarize_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Initialize a list to store summarized sentences
    summarized_sentences = []
    
    # Process each sentence for summarization
    for sentence in sentences:
        words = TextBlob(sentence.lower()).words
        
        # Filter out stopwords from the words
        filtered_words = [word for word in words if word not in stop_words]
        
        # Join the filtered words to form a summarized sentence
        summarized_sentence = ' '.join(filtered_words)
        
        # Append the summarized sentence to the list
        summarized_sentences.append(summarized_sentence)
    
    # Join the summarized sentences to form the final summarized text
    summarized_text = ' '.join(summarized_sentences)
    
    return summarized_text

def translate_text_to_tamil(text):
    translator = Translator()
    translation = translator.translate(text, src='en', dest='ta')
    return translation.text

def retrieve_text_from_image(image_path):
    pytesseract.pytesseract.tesseract_cmd = r"C:/Users/Lenovo/Desktop/speech based transaltion - complete project/Tesseract-OCR/tesseract.exe"
    actual_image = cv2.imread(image_path)
    cropped_image = actual_image[0:800, 0:1000]
    sample_img = cv2.resize(cropped_image, (400, 450))
    image_ht, image_wd, image_thickness = sample_img.shape
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    texts = pytesseract.image_to_string(sample_img)
    return texts

UPLOAD_FOLDER = 'uploads'  # Define your upload folder
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'} 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file.filename.endswith(('txt', 'pdf')):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            with open(file_path, 'r', encoding='utf-8') as file_content:
                text = file_content.read()
        # If file is an image file
        elif file.filename.endswith(('png', 'jpg', 'jpeg')):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            text = retrieve_text_from_image(file_path)
        summarized_text = summarize_text(text)
        translated_text = translate_text_to_tamil(summarized_text)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'translated_audio.mp3')
        convert_to_audio(translated_text, save_path)
        translated_audio_url = '/get_audio'  # URL to access the translated audio file
        return render_template('result.html', summarized_text=summarized_text, translated_text=translated_text, translated_audio_url=translated_audio_url)

def convert_to_audio(text, save_path):
    tts = gTTS(text, lang='ta')
    tts.save(save_path)

@app.route('/get_audio')
def get_audio():
    return send_file('uploads/translated_audio.mp3', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False, port=800)
