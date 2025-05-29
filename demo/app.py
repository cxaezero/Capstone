from flask import Flask, Response, render_template
import cv2
import datetime

app = Flask(__name__)

def generate_stream(key):
    cap = cv2.VideoCapture(f"rtmp://localhost:1935/live/{key}")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/stream/<key>')
def stream_video(key):
    return Response(generate_stream(key), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def dashboard():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    location = "Seoul, South Korea"
    return render_template("dashboard.html", time=now, location=location)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
