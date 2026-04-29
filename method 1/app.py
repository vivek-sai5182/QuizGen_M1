import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from backend.pdf_reader import extract_text
from backend.rule_based_qmaker import generate_mcqs

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    num_questions = int(request.form.get("num_questions", 15))

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    text = extract_text(file_path)
    if not text or text == "Unsupported file format":
        return jsonify({"error": "Could not extract text from file"}), 400

    mcqs = generate_mcqs(text, num_questions=num_questions)

    if not mcqs:
        return jsonify({"error": "Could not generate questions from this file. Try a different document."}), 400

    return jsonify({"questions": mcqs, "total": len(mcqs)})


if __name__ == "__main__":
    app.run(debug=True)