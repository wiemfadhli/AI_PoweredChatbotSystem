from flask import Flask, request, jsonify
from flask_cors import CORS

from rag_query import ask_question   # 👈 import your function

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"reply": "Please enter a message"})

    # 🔥 CALL YOUR RAG SYSTEM
    answer = ask_question(user_message)

    return jsonify({
        "reply": answer
    })

if __name__ == "__main__":
    app.run(debug=True)