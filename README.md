# Real-Time Football Analysis with Multi-Model AI Commentary
## 1. Project Overview
This project presents a state-of-the-art computer vision pipeline for real-time football match analysis. It combines high-performance object tracking with a powerful, multi-model AI commentary system to generate live, insightful text descriptions of the game.

The system processes video footage to detect, track, and assign players to their respective teams. This rich contextual data, including player positions and camera motion, is then fed into a suite of cutting-edge multimodal Large Language Models (LLMs)—Google Gemini, Salesforce BLIP, and Qwen-VL—to produce automated, human-like analysis directly on the video output.

## 2. Key Features
High-Performance Tracking: Utilizes YOLOv8 for robust, real-time detection and tracking of players, goalkeepers, and the ball.

Automated Team Assignment: Employs K-Means clustering on jersey colors to automatically classify players into two distinct teams.

Tactical 2D Mini-Map: Renders a dynamic overhead view of all player positions on a 2D pitch for tactical analysis.

Multi-Model AI Commentary: Integrates several advanced LLMs to provide diverse, real-time textual analysis of the on-field action.

Camera Motion Compensation: Estimates homography to stabilize the field of view, ensuring accurate player mapping even with dynamic camera movement.

Rich Visualization: Produces an annotated video stream with color-coded bounding boxes, player IDs, the tactical mini-map, and the AI-generated commentary.

## 3. How It Works: The Analysis Pipeline
Video Input: The system takes a football match video as input.

Computer Vision Processing: Each frame is processed to:

Detect players, the ball, and goalkeepers using YOLOv8.

Track each object across frames, assigning a persistent ID.

Assign players to teams based on jersey color.

Calculate the camera's perspective transformation (homography).

AI Commentary Generation: The processed frame, rich with object location data, is passed as input to one of the integrated multimodal LLMs (Gemini, BLIP, or Qwen-VL). The model interprets the visual information and generates a concise, descriptive text caption of the current game state.

Annotated Output: The final video is rendered with all visual overlays and the AI-generated text, creating a comprehensive analytical tool.

## 4. Core Technologies
This project is built on a powerful stack of computer vision and generative AI technologies:

Object Detection & Tracking: YOLOv8, Custom Tracking Logic

In-Game Analytics: K-Means Clustering, OpenCV for Homography

AI-Powered Commentary (Image-to-Text):

Google Gemini: For detailed, context-aware tactical analysis.

Salesforce BLIP: For generating concise, descriptive captions.

Qwen-VL: For understanding and describing complex visual scenes.

Core Libraries: PyTorch, Transformers, OpenCV, Supervision, Scikit-learn, NumPy, mplsoccer.

## 5. Setup and Installation
A Python environment with GPU support is highly recommended for optimal performance.

Clone or download the project files.

Install the necessary packages. The notebook contains all the required pip installation commands.

# Core computer vision and data science libraries
pip install ultralytics supervision numpy opencv-python scikit-learn pandas mplsoccer

# For running the multimodal LLMs
pip install transformers torch torchvision google-generativeai

# Ensure all packages are up-to-date
pip install --upgrade ultralytics torch torchvision transformers

6. Execution Guide
Launch the Jupyter Notebook: Open the player-tracking10 (3).ipynb file in an environment like Jupyter Lab or Google Colab.

Configure Paths and API Keys:

Set the correct paths for the input video file and the trained YOLOv8 model (best.pt).

For Gemini, ensure your Google AI API key is correctly configured within the notebook.

Run the Cells: Execute the notebook cells in sequential order. The script will initialize all models and then begin processing the video frame-by-frame, generating the final annotated video file as the output.

