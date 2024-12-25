# Real-Time Face, Emotion, Age, and Gender Detection

This project implements a real-time system to detect faces, classify emotions, estimate age, and determine gender using OpenCV and pre-trained deep learning models.

## 🚀 Features
- **Face Detection**: Utilizes Haar Cascade Classifier for real-time face detection.
- **Emotion Detection**: Classifies emotions into categories like Angry, Happy, Sad, etc.
- **Age Estimation**: Predicts the approximate age of the detected person.
- **Gender Classification**: Classifies the detected person as Male or Female.
- **Real-Time Processing**: Processes video input from a webcam.

## 📦 Requirements
- Python 3.x
- OpenCV
- Keras
- TensorFlow
- NumPy

## 📂 Pretrained Models
- **Haar Cascade Classifier**: For face detection.
  - `pretrained_haarcascade_classifier/haarcascade_frontalface_default.xml`
- **Emotion Detection Model**: Keras model for emotion classification.
  - `models/emotion_detection_model.h5`
- **Age Prediction Model**: Keras model for age estimation.
  - `models/age_model.h5`
- **Gender Classification Model**: Keras model for gender prediction.
  - `models/gender_classification_model.h5`

## 🛠 Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repository.git
   cd your-repository
   ```
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the pre-trained models in the `models/` directory.
4. Place the Haar Cascade XML file in the `pretrained_haarcascade_classifier/` directory.

## ▶️ Usage
Run the script to start the real-time detection:
```bash
python main.py
```
Press `q` to exit the video feed.

## 📚 Explanation
1. **Face Detection**:
   - Detects faces in the video frame using the Haar Cascade Classifier.
   - Extracts the region of interest (ROI) for further processing.

2. **Emotion Detection**:
   - Scales the grayscale ROI to 48x48 pixels.
   - Normalizes and preprocesses the image for emotion prediction.
   - Outputs a label such as `Happy`, `Sad`, or `Angry`.

3. **Gender Classification**:
   - Resizes the ROI to 244x244 pixels.
   - Normalizes the pixel values and predicts gender.

4. **Age Estimation**:
   - Resizes the ROI to 200x200 pixels.
   - Predicts the age as a rounded integer.

5. **Visualization**:
   - Draws rectangles around detected faces.
   - Displays labels for emotion, gender, and age on the video feed.

## 🔧 Customization
- **Add more emotions**: Extend the `class_labels` array to include additional emotion categories.
- **Fine-tune models**: Replace the pre-trained models with your custom-trained ones.
- **Adjust Haar Cascade**: Modify the parameters (e.g., `scaleFactor` and `minNeighbors`) in `detectMultiScale` for better face detection.

## 🤝 Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue.
