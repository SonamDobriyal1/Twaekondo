from flask import Flask, render_template, request, jsonify
import asyncio
import cv2
import os
import numpy as np
import json
import logging
from threading import Thread
from aiohttp import web, WSMsgType
from aiohttp_cors import setup as cors_setup, ResourceOptions
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration
from aiortc.contrib.media import MediaRelay
from pose_detector import PoseDetector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
relay = MediaRelay()

# Load poses when the application starts
with app.app_context():
    tutorial_poses = load_tutorial_poses()

class PoseProcessingTrack(VideoStreamTrack):
    """
    A video stream track that transforms frames by applying pose detection.
    Optimized for low latency by processing every 3rd frame only.
    """

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_count = 0
        self.last_keypoints = None
        self.last_similarity = 0
        self.last_has_pose = False
        self.skip_frames = 2  # Process every 3rd frame for better performance

    async def recv(self):
        frame = await self.track.recv()
        
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Process only every 3rd frame to reduce latency, but always draw overlays
        self.frame_count += 1
        if self.frame_count % (self.skip_frames + 1) == 0:
            # Process frame using pose detector
            global current_pose_index
            if current_pose_index < len(tutorial_poses):
                processed_img, similarity, has_pose, best_frame = pose_detector.process_frame(
                    img,
                    tutorial_poses[current_pose_index]['landmarks']
                )
                # Store the latest pose data
                self.last_keypoints = pose_detector.current_keypoints
                self.last_similarity = similarity
                self.last_has_pose = has_pose
            else:
                processed_img, similarity, has_pose, _ = pose_detector.process_frame(img)
                self.last_keypoints = pose_detector.current_keypoints
                self.last_similarity = similarity
                self.last_has_pose = has_pose
        else:
            # Use cached pose data to draw consistent overlays without full processing
            if self.last_keypoints is not None and self.last_has_pose:
                processed_img = pose_detector.draw_pose_overlay(
                    img, 
                    self.last_keypoints, 
                    self.last_similarity
                )
            else:
                processed_img = img
        
        # Convert back to frame
        result_frame = frame.from_ndarray(processed_img, format="bgr24")
        result_frame.pts = frame.pts
        result_frame.time_base = frame.time_base
        
        return result_frame

# Store peer connections
pcs = set()

async def offer(request):
    """Handle WebRTC offer with low latency configuration"""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Create peer connection with simple configuration
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track {track.kind} received")
        if track.kind == "video":
            # Transform the video track with relay for better performance
            relayed_track = relay.subscribe(track)
            transformed_track = PoseProcessingTrack(relayed_track)
            pc.addTrack(transformed_track)

    # Handle the offer
    await pc.setRemoteDescription(offer)
    
    # Create answer with low latency preferences
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

async def check_pose_match_handler(request):
    """Check if current pose matches target pose"""
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
                return web.json_response({'matched': True, 'similarity': similarity})
    return web.json_response({'matched': False, 'similarity': 0.0})

async def update_pose_index_handler(request):
    """Update the current pose index"""
    global current_pose_index
    try:
        index = int(request.match_info['index'])
        if 0 <= index < len(tutorial_poses):
            current_pose_index = index
            return web.json_response({'success': True, 'current_index': index})
        return web.json_response({'success': False, 'error': 'Invalid index'})
    except (ValueError, KeyError):
        return web.json_response({'success': False, 'error': 'Invalid index format'})

async def init_aiohttp_app():
    """Initialize aiohttp application"""
    aiohttp_app = web.Application()
    
    # Setup CORS
    cors = cors_setup(aiohttp_app, defaults={
        "*": ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Add routes
    aiohttp_app.router.add_post("/offer", offer)
    aiohttp_app.router.add_get("/check_pose_match", check_pose_match_handler)
    aiohttp_app.router.add_get("/update_pose_index/{index}", update_pose_index_handler)
    
    # Add CORS to all routes
    for route in list(aiohttp_app.router.routes()):
        cors.add(route)
    
    return aiohttp_app

# Flask routes
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

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/')
def index():
    return render_template('index.html')

def run_aiohttp_server():
    """Run aiohttp server in a separate thread"""
    async def run_server():
        aiohttp_app = await init_aiohttp_app()
        runner = web.AppRunner(aiohttp_app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", 8080)
        await site.start()
        logger.info("WebRTC server started on http://0.0.0.0:8080")
        
        # Keep the server running
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            pass
        finally:
            # Cleanup
            for pc in pcs:
                await pc.close()
            await runner.cleanup()
    
    # Run in new event loop in thread
    def thread_target():
        asyncio.set_event_loop(asyncio.new_event_loop())
        asyncio.run(run_server())
    
    thread = Thread(target=thread_target, daemon=True)
    thread.start()
    return thread

if __name__ == '__main__':
    # Start WebRTC server
    webrtc_thread = run_aiohttp_server()
    
    # Start Flask server
    app.run("0.0.0.0", port=10000, debug=False)