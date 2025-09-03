from flask import Flask, render_template, Response, jsonify
import cv2
import os
from pose_detector import PoseDetector

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/poses')

# Create static/poses directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    

# Initialize pose detector with higher threshold
pose_detector = PoseDetector(similarity_threshold=90.0)

def load_tutorial_poses():
    poses = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for img_file in sorted(os.listdir(app.config['UPLOAD_FOLDER'])):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file)
                keypoints = pose_detector.get_pose_landmarks(img_path)
                # Convert keypoints to regular numpy array if they exist
                if keypoints is not None and len(keypoints) > 0:
                    keypoints = keypoints.copy()  # Make a copy to ensure we have our own data
                    poses.append({
                        'landmarks': keypoints,
                        'filename': img_file
                    })
                    print(f"Loaded pose image: {img_file}")
    return poses

# Initialize variables
tutorial_poses = []
current_pose_index = 0

# Load poses when the application starts
with app.app_context():
    tutorial_poses = load_tutorial_poses()

@app.route('/update_pose_index/<int:index>')
def update_pose_index(index):
    global current_pose_index
    if 0 <= index < len(tutorial_poses):
        current_pose_index = index
        return jsonify({'success': True, 'current_index': index})
    return jsonify({'success': False, 'error': 'Invalid index'})

@app.route('/check_pose_match')
def check_pose_match():
    global current_pose_index
    if current_pose_index < len(tutorial_poses):
        if pose_detector.current_keypoints is not None:
            similarity = pose_detector.calculate_pose_similarity(
                pose_detector.current_keypoints,
                tutorial_poses[current_pose_index]['landmarks']
            )
            # Convert numpy float32 to Python float
            similarity = float(similarity)
            if similarity >= pose_detector.similarity_threshold:
                return jsonify({'matched': True, 'similarity': similarity})
    return jsonify({'matched': False, 'similarity': 0.0})

def generate_frames():
    # Try different backends
    backends = [
        cv2.CAP_DSHOW,  # DirectShow (Windows)
        cv2.CAP_MSMF,   # Microsoft Media Foundation
        cv2.CAP_ANY     # Auto-detect
    ] 
    
    camera = None
    for backend in backends:
        print(f"Trying camera with backend: {backend}")
        camera = cv2.VideoCapture(0 + backend)
        
        if camera.isOpened():
            print(f"Successfully opened camera with backend {backend}")
            # Set camera properties
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            # Try to read a test frame
            ret, _ = camera.read()
            if ret:
                print("Successfully read test frame")
                break
            else:
                print("Could not read frame, trying next backend")
                camera.release()
        else:
            print(f"Failed to open camera with backend {backend}")
    
    if camera is None or not camera.isOpened():
        print("Error: Could not open any camera")
        return

    # Set camera properties
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)


    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Error: Could not read frame")
                break
                
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)

            # Process frame using pose detector
            if current_pose_index < len(tutorial_poses):
                frame, similarity, has_pose, best_frame = pose_detector.process_frame(
                    frame,
                    tutorial_poses[current_pose_index]['landmarks']
                )
            else:
                frame, _, has_pose, _ = pose_detector.process_frame(frame)

            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        camera.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run("0.0.0.0", port=10000)