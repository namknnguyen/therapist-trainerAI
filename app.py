import random
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from textblob import TextBlob

app = Flask(__name__)
# Add explicit CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173"],  # Your frontend origin
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

def call_ollama(prompt: str, model: str = "llama3.2:3b") -> str:
    """
    Calls the LLM via Ollama using the specified model with the given prompt.
    Handles errors more gracefully and allows for model specification.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            check=True,
            timeout=30  # Add a timeout to prevent indefinite hangs
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error calling LLM ({model}): {e}")
        return f"LLM error: {e}" #More informative error message
    except subprocess.TimeoutExpired:
        print(f"LLM call timed out for model {model}")
        return "LLM request timed out."


@app.route('/')
def home():
    return "Welcome! The server is running on port 8000."

@app.route('/start', methods=['POST'])
def start_session():
    data = request.get_json()
    situation = data.get('situation')
    personality = data.get('personality')
    style = data.get('style')
    role = data.get('role')

    if not all([situation, personality, style, role]):
        return jsonify({'error': 'Missing parameters'}), 400

    initial_prompt = (
        f"You are a patient experiencing {situation.lower()}. "
        f"Your personality is {personality.lower()}. "
        f"You are engaging in a {style} conversation with a {role}. "
        "Please respond as the patient, conveying your feelings and thoughts authentically. Only output the response, nothing else."
        f"while maintaining a {personality.lower()} personality style.  "
        "Remember your role as the patient and focus on your immediate emotional response."
    )

    patient_response = call_ollama(initial_prompt)
    conversation_history = [
        {"speaker": "Patient", "text": patient_response}
    ]
    return jsonify({
        'response': patient_response,
        'conversation_history': conversation_history,
        'situation': situation,
        'personality': personality,
        'style': style,
        'role': role
    })

@app.route('/message', methods=['POST'])
def message():
    try:
        data = request.get_json()
        conversation_history = data.get('conversation_history', [])
        user_message = data.get('message')
        situation = data.get('situation')
        personality = data.get('personality')
        style = data.get('style')
        role = data.get('role')

        # Add user message to history
        conversation_history.append({"speaker": "You", "text": user_message})

        # Generate patient response
        conversation_context = "\n".join(f"{entry['speaker']}: {entry['text']}" for entry in conversation_history)
        prompt = (
            f"The following is a conversation between a {role} and a patient. "
            f"As the patient, experiencing {situation.lower()}, with a {personality.lower()} personality, and communicating in a {style} manner, respond only with your next statement and nothing else:\n\n"
            f"{conversation_context}"
        )
        patient_response = call_ollama(prompt)
        
        # Analyze the patient's response mood using TextBlob
        analysis = TextBlob(patient_response)
        mood_score = analysis.sentiment.polarity
        
        # Classify mood based on the sentiment polarity
        if mood_score > 0.3:
            mood = 'Positive'
        elif mood_score < -0.3:
            mood = 'Negative'
        else:
            mood = 'Neutral'
        
        # Append the response with mood info
        conversation_history.append({
            "speaker": "Patient",
            "text": patient_response,
            "mood": mood,
            "mood_score": mood_score
        })

        # Evaluate therapeutic quality (as before)
        eval_prompt = (
            f"Evaluate the therapeutic quality of this {role}'s response: '{user_message}'.\n"
            "Provide a brief, less than 20 word evaluation focusing on its helpfulness and appropriateness. Suggest a better response."
        )
        eval_text = call_ollama(eval_prompt).strip()

        # Determine color based on evaluation sentiment (if needed)
        eval_analysis = TextBlob(eval_text)
        polarity = eval_analysis.sentiment.polarity
        if polarity > 0.3:
            color_response = 'GREEN'
        elif polarity > -0.3:
            color_response = 'YELLOW'
        else:
            color_response = 'RED'

        response_data = {
            'conversation_history': conversation_history,
            'response': patient_response,
            'evaluation': {
                'color': color_response,
                'text': eval_text
            }
        }
        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'conversation_history': conversation_history,
            'evaluation': {
                'color': 'RED',
                'text': f'Error: {str(e)}'
            }
        }), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    conversation_history = data.get('conversation_history', [])

    if not conversation_history:
        return jsonify({'error': 'No conversation history provided'}), 400

    conversation_text = "\n".join(f"{entry['speaker']}: {entry['text']}" for entry in conversation_history)
    eval_prompt = (
        f"Review the following therapy session conversation:\n{conversation_text}\n"
        "Provide a constructive evaluation and feedback of the session, noting strengths and areas for improvement.  "
        "Be specific and concise."
    )
    evaluation = call_ollama(eval_prompt)
    return jsonify({
        'evaluation': evaluation,
    })

if __name__ == '__main__':
    app.run(debug=True, port=8000)