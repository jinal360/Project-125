from flask import Flask
import jsonify

app= Flask(__name__)

@aap.route("/predict-alphabet",methods=["POST"])
def persict_data():
    image = request.files.get("alphabet")
    perdiction = get_perdiction(image)
    return jsonify({
        "perdiction": perdiction
    }),200