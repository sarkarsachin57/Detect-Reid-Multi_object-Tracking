# Detect-Reid-Multi_object-Tracking
This is a custom detection followed by reidentification based multi object tracking algorithm. 

### Get Started

**Installation**
```
pip install -r requirements.txt
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

**Run Tracking on Video**
```
cd Detect-Reid-Multi_object-Tracking
python reid_track.py --input-path "demo_videos\demo_video.webm"
```

**Run Tracking on live stream or webcam**
```
cd Detect-Reid-Multi_object-Tracking
python reid_track.py --input-path 0 --is-stream
```
