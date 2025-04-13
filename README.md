# Down Syndrome Detector

This project is a deep learning-based application that detects Down syndrome from images using a pre-trained model and face detection. The application uses the **MTCNN** face detection model and a custom-trained CNN model to predict whether a person has Down syndrome or is healthy.

## Features
- **Face Detection**: Detects faces in the uploaded images using MTCNN.
- **Prediction**: Classifies the face as either "Down Syndrome" or "Healthy" with a confidence score.
- **GUI Interface**: A simple Tkinter-based graphical user interface (GUI) that allows users to upload images, detect conditions, and display results with bounding boxes around detected faces.

## Requirements

- Python 3.12 or later
- TensorFlow (for the pre-trained model)
- MTCNN (for face detection)
- Tkinter (for GUI)
- OpenCV (for image processing)
- Pillow (for image handling)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/down-syndrome-detector.git
    cd down-syndrome-detector
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/Mac
    venv\Scripts\activate     # For Windows
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download or place your dataset folder containing the `downsyndrome/` and `healthy/` directories with respective images inside the `dataset/` directory. Your folder structure should look like this:
    ```
    dataset/
    ├── downsyndrome/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── healthy/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ```

5. Run the application:
    ```bash
    python app.py
    ```

## Usage
- Upload an image by clicking the **Upload Image** button.
- The face will be detected, and the prediction will be displayed with a confidence score.
- A bounding box will be drawn around the detected face.

## Model Information
The model used for prediction is a custom CNN model trained on a dataset containing images classified into two categories:
- **Down Syndrome**
- **Healthy**

## Contributing
Feel free to fork this project and contribute by opening issues or submitting pull requests. Just be sure to credit the original author and follow the conditions outlined in the repository.

## License
This project is open-source.
