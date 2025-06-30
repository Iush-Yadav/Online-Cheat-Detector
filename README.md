**Cheat Detection Simple** - Python code to detect left-right eye movement

**Cheat Detection App and Web** - Python and Html code to work on system background and show logs on web dashboard

# 📚 Online Exams Cheat Detection System

This project implements a real-time **eye movement-based cheat detection system** for online exams. Using Mediapipe’s iris tracking, it detects suspicious behaviors like frequent or prolonged glances away from the screen — helping maintain integrity during remote tests.

---

## 🚀 Features

✅ Real-time eye gaze tracking with Mediapipe Face Mesh  
✅ Works independently of head movement — focuses on actual eye movement  
✅ Calibration phase for personalized neutral gaze  
✅ Counts left and right glances  
✅ Detects prolonged glances to the sides  
✅ Audible warnings (beeps) when suspicious thresholds are crossed  
✅ Simple OpenCV interface with live feedback  
✅ Easy to set up and run locally

---

## 🎯 Use Case

Designed for **remote proctoring**: the system monitors examinee eye movements to identify behaviors consistent with cheating (e.g., looking off-screen repeatedly or for long periods).

---

## 📦 Requirements

- Python
- Flask
- OpenCV
- Mediapipe
