# 📷 AI Security System: Face Recognition & Object Detection

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange?style=for-the-badge&logo=yolo)
![Flask](https://img.shields.io/badge/Flask-Web-lightgrey?style=for-the-badge&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-green?style=for-the-badge&logo=opencv)

A comprehensive real-time security dashboard powered by AI. This system performs **Face Recognition** to identify specific individuals and **Object Detection** (using YOLOv8) to identify items like mobile phones, bottles, and bags simultaneously.

---

## 🚀 Key Features

### 👤 Face Recognition
* **Real-time Identification:** Detects faces and matches them against a watchlist.
* **Red Alert System:** Triggers a visual **RED ALERT** if a watchlist person is detected.
* **Visitor Counting:** Counts human faces in real-time.
* **Auto-Learning:** Upload a photo via the dashboard, and the system learns the face instantly without restarting.

### 📦 Object Detection (New!)
* **Powered by YOLOv8:** Uses the state-of-the-art YOLO model for ultra-fast detection.
* **Item Recognition:** Identifies common objects like **Phones, Laptops, Bottles, Bags, etc.**
* **Performance Optimized:** Uses frame skipping and resizing to run smoothly on standard CPUs.

---

## 🛠️ Tech Stack

* **Core:** Python 3.11
* **Web Framework:** Flask
* **AI Models:** * `face_recognition` (dlib) for Faces
  * `ultralytics` (YOLOv8) for Objects
* **Database:** SQLite3 (for logging alerts)

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name