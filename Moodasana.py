from flask import Flask, request, render_template, redirect, url_for, session
from transformers import pipeline
from deepface import DeepFace
import nltk 
import os

nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = 'super_secret_key'  
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

def analyze_sentiment(text):
  
    result = sentiment_pipeline(text)[0]
    label = result.get('label', 'NEUTRAL') 
    sentiment_scores = {
        'POSITIVE': 1,
        'NEGATIVE': -1,
        'NEUTRAL': 0  
    }
    return sentiment_scores.get(label.upper(), 0)



#  Face Sentiment
def analyze_face(image_path):
   
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        
        # Handle if result is a list (DeepFace sometimes returns a list of results)
        if isinstance(result, list) and result:
            result = result[0]  # Use the first dictionary from the list
        
        # Ensure 'emotion' key exists
        if 'emotion' not in result:
            raise ValueError("Emotion data not found in DeepFace analysis result.")
        
        # Assign scores based on emotions
        emotion_scores = {
            'angry': -2,
            'disgust': -2,
            'fear': -1,
            'sad': -1,
            'neutral': 0,
            'happy': 2,
            'surprise': 2
        }

        # Determine the dominant emotion
        dominant_emotion = max(result['emotion'], key=result['emotion'].get)
        return emotion_scores.get(dominant_emotion.lower(), 0)  # Default to 0 for unexpected emotions
    except Exception as e:
        print(f"Error analyzing face sentiment: {e}")
        return 0
    


@app.route('/')
def home():
    session.clear()  # Clear previous data
    session['final_score'] = 0  # Initialize score within a valid request context
    return render_template('welcome.html')

@app.route('/survey1', methods=['GET', 'POST'])
def survey1():
    if request.method == 'POST':
        responses = request.form.getlist('response')
        score = sum([int(resp) for resp in responses])
        session['survey1_score'] = score
        return redirect(url_for('survey2'))
    # Sample questions
    questions = [
        "How are you feeling today?",
        "How was your sleep last night?",
        "How satisfied are you with your work-life balance?",
        "How do you feel about your physical health?",
        "How often do you feel stressed?",
        "How would you rate your happiness overall?"
    ]
    return render_template('survey1.html', questions=questions)

@app.route('/survey2', methods=['GET', 'POST'])
def survey2():
    if request.method == 'POST':
        if 'skip' in request.form:
            session['survey2_score'] = 0  # Neutral score for skip
            return redirect(url_for('survey3'))
        
        text = request.form.get('message', '').strip()
        if text:
            score = analyze_sentiment(text)
            session['survey2_score'] = score
            return redirect(url_for('survey3'))
        else:
            return render_template('survey2.html', error="Please enter some text or skip.")
    return render_template('survey2.html')


@app.route('/survey3', methods=['GET', 'POST'])
def survey3():
    if request.method == 'POST':
        if 'skip' in request.form:
            session['survey3_score'] = 0  # Neutral score for skip
            return redirect(url_for('results'))
        
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file and file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            score = analyze_face(file_path)
            session['survey3_score'] = score
            return redirect(url_for('results'))
    return render_template('survey3.html')


@app.route('/results')
def results():
    # Sum the scores from the different surveys
    final_score = sum([
        session.get('survey1_score', 0),
        session.get('survey2_score', 0),
        session.get('survey3_score', 0)
    ])

    # Determine sentiment based on the final score
    sentiment = 'Positive' if final_score > 0 else 'Negative' if final_score < 0 else 'Neutral'

    # Pass the sentiment and score to the results template
    return render_template('results.html', final_score=final_score, sentiment=sentiment)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(port=2000, debug=True)
