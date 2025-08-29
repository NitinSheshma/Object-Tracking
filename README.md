# 🎯 Object-Tracking  

A real-time **object tracking project** using **Python, OpenCV, and YOLOv8**.  
This project captures video from your webcam and uses Ultralytics YOLOv8 + ByteTrack to detect and track multiple objects with unique IDs in real-time.  

---

## 🚀 Features
- Real-time object detection and tracking via **YOLOv8**  
- Assigns **unique IDs** to objects with **ByteTrack**  
- Works with **webcam** or video files  
- Easy to set up and run on any system  

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NitinSheshma/Object-Tracking.git
   cd Object-Tracking
(Optional but recommended) Create a virtual environment:

bash
Copy code
python -m venv venv
venv\Scripts\activate   # On Windows
Install requirements:

bash
Copy code
pip install -r requirements.txt
▶️ Usage
Run the tracking script with your webcam:

bash
Copy code
python yolo_webcam_tracking.py
Press q to quit the live stream.

Change CAM_INDEX in the script if you have multiple cameras.

You can also modify the script to use a video file instead of a webcam.

📂 Project Structure
bash
Copy code
Object-Tracking/
│── yolo_webcam_tracking.py   # Main script for YOLOv8 + webcam tracking
│── requirements.txt          # Dependencies
│── README.md                 # Project description
🛠️ Tech Stack
Python 3.10+

OpenCV

Ultralytics YOLOv8

PyTorch

NumPy

📸 Example Output
Here’s how the tracking looks in action (bounding boxes + IDs):


(You can replace example.png with your own screenshot or GIF from a tracking run)

🤝 Contributing
Feel free to fork this repo, open issues, or submit pull requests. Contributions are welcome!

📜 License
This project is licensed under the MIT License – you are free to use, modify, and distribute it.

yaml
Copy code

---
#snapshots
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/1a53fc67-a3b2-4dfb-8639-3ed3356521c7" />


