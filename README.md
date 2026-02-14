# 📷 Intelligent Face Recognition & Security System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey?style=for-the-badge&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)


A robust, real-time Web CCTV and Face Recognition system built with Python 3.11. This system is designed to detect faces, count visitors in real-time, and trigger **RED ALERTS** when a specific person from the watchlist is identified.

---

## 🚀 Key Features

* **Real-time Face Detection:** Uses advanced AI to detect and track faces via webcam.
* **Watchlist Alert System:** Upload a photo to the system, and it will instantly trigger a **RED ALERT** if that person appears on camera.
* **Live Visitor Counting:** Accurate real-time counting of people in the frame.
* **Crash-Proof Image Handling:** Integrated with OpenCV for safe image processing and upload stability.
* **Auto-Updating Dashboard:** Live logs update automatically without refreshing the page (AJAX).
* **Database Logging:** Detection history is saved to a SQLite database.

---

## 🛠️ Tech Stack

* **Language:** Python 3.11 (Strictly Recommended)
* **Backend:** Flask
* **Computer Vision:** OpenCV (`cv2`), `face_recognition`, `dlib`
* **Frontend:** HTML5, Bootstrap 5, JavaScript
* **Database:** SQLite3

---

## ⚙️ Installation & Setup

This project is optimized for **Python 3.11**. Follow these steps to clone and run the project.

### 1. Clone the Repository
Open your terminal or command prompt and run:

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name