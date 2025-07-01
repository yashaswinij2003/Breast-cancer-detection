#launch GUI
from flask import Flask, request, redirect
import subprocess

app = Flask(__name__)

@app.route("/", methods=["POST"])
def run_classifier():
    username = request.form.get("uname", "Unknown User")
    email = request.form.get("ema", "No Email")

    print(f"Received Data -> Username: {username}, Email: {email}")  # Debugging

    # Run GUI
    subprocess.Popen(["python", "gui_classifier.py"])  # Runs Tkinter GUI

    # Redirect to confirmation page
    return redirect("http://127.0.0.1:5000/success")

@app.route("/success")
def success():
    return "<h1>GUI has been launched!</h1><p>Please check the new window.</p>"

if __name__ == "__main__":
    app.run(debug=True)