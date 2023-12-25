from llm_inference import query_llm
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/query', methods=['POST'])
def query_endpoint():
    if request.method == 'POST':
        try:
            # Get user prompt from HTML page
            user_prompt = request.form['user_prompt']
            
            # Query LLM
            response = query_llm(user_prompt)

        except Exception as e:
            # return render_template('index.html', user_prompt=user_prompt, error=str(e))
            return "error:{}".format(e)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888, debug=True)
