from flask import Flask, jsonify, send_from_directory
import subprocess
import os

app = Flask(__name__)

@app.route('/run-utilization-script', methods=['GET'])
def run_script():
    try:
        # Run the Python script
        result = subprocess.run(["python", "Plot.py"], capture_output=True, text=True)

        # Print output in Flask logs
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        return jsonify({
            "status": "success",
            "images": [
                "/static/memory_vs_disk.png",
                "/static/uptime_vs_disk.png",
                "/static/correlation_heatmap.png"
            ]
        })

    except subprocess.CalledProcessError as e:
        print("Error Running Plot.py:", e.stderr)
        return jsonify({"status": "error", "error": e.stderr})

    except Exception as e:
        print("Unexpected Error:", str(e))
        return jsonify({"status": "error", "error": str(e)})

# Serve images from static folder
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory("static", filename)

if __name__ == '__main__':
    app.run(debug=True)
