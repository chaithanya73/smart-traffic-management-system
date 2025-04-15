from flask import Flask, render_template
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    junction = "Main Junction"
    current_time = datetime.now().strftime("%H:%M:%S")
    diversions = [
        {"destination": "Airport", "via": "East Blvd → Central Rd"},
        {"destination": "Mall", "via": "Hill Rd → Park Lane"},
        {"destination": "Tech Park", "via": "Loop St → North Ave"},
    ]
    return render_template("index.html", junction=junction, time=current_time, diversions=diversions)

if __name__ == "__main__":
    app.run(debug=True)
