# DroidCore
## Iterations
Astro_x1: 5D2: BD-2-X5 - Current 


<img width="1007" height="809" alt="Screenshot 2025-07-27 131934" src="https://github.com/user-attachments/assets/94c8076f-001c-422e-be90-e081c3d5f513" />
<img width="873" height="781" alt="Screenshot 2025-07-27 131946" src="https://github.com/user-attachments/assets/bb4de529-28dc-46ef-9cca-74b725fca561" />
<img width="839" height="724" alt="Screenshot 2025-07-27 131958" src="https://github.com/user-attachments/assets/5f149801-9d8e-4ba2-8bd3-dadccc7a1883" />


## Concept
<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/ceee8bcf-c80d-4f1a-91e9-343fe5df77cd" />


A project focused on building an advanced robotic platform incorporating multimodal AI capabilities (Vision, Speech, Language) and low-level hardware control.

## Features

### High-Level AI & Perception
*   **Speech Processing:** Utilizes Speech-to-Text (e.g., Whisper) and Text-to-Speech (e.g., Piper, Orpheus). Includes sound synthesis capabilities (ggwave).
*   **Language Understanding:** Leverages Large Language Models (e.g., Ollama + DeepSeek R1 7B) for reasoning, conversation, and potentially task execution.
*   **Computer Vision:** Integrates vision capabilities using sensors like Kinect v2 (via libfreenect2), employing techniques like SLAM, Object Detection (YOLO), and Face Recognition (OpenFace/MediaPipe).

### Low-Level Hardware Control & Sensing
*   **Motor Control:** Implements advanced motor control strategies like Field-Oriented Control (FOC).
*   **Sensing:** Incorporates various sensors, including Ultrasound.
*   **Communication:** Utilizes communication protocols like Bluetooth.
*   **Actuation:** Controls hardware components like Relays, fans.

## Architecture Overview

The system follows a layered architecture:
*   **`HighLvl`:** Manages complex AI tasks, perception, decision-making, and human interaction. Potential use of middleware like ROS.
    *   Integrates Vision, Language, Speech, and Sound processing modules.
*   **`LowLvl`:** Handles direct hardware interfacing, real-time control, and sensor data acquisition.
    *   Includes drivers and control logic for motors (FOC), sensors (Radar), communication hardware (Bluetooth), and other peripherals.

## Directory Structure

```
├── HighLvl/
│   ├── Vision/       # Vision processing, SLAM, Object Detection, Face Recognition
│   ├── Language/     # Language models, Reasoning, Text Generation
│   ├── Speech/       # Speech-to-Text (STT) and Text-to-Speech (TTS)
│   └── Sound/        # Sound synthesis (ggwave), potentially audio processing
├── LowLvl/
│   ├── Fans/         # Control for cooling or other fan systems
│   ├── Radar/        # Radar sensor interface and data processing
│   ├── Motor/        # Motor drivers and control logic
│   ├── FOC/          # Field-Oriented Control specific implementations
│   ├── Build/        # Build system files/scripts for low-level firmware
│   ├── Bluetooth/    # Bluetooth communication stack/interface
│   ├── Base/         # Core low-level functionalities, drivers, or base platform code
│   └── BleFF/        # Potentially Bluetooth Low Energy related functionality
└── README.md         # This file
```

## Core Technologies

*   **AI/ML:** Ollama, DeepSeek, OpenAI Whisper, Piper TTS, YOLO, OpenFace, MediaPipe
*   **Vision:** libfreenect2 (Kinect v2)
*   **Middleware:** ROS (planned)
*   **Hardware Control:** Field-Oriented Control (FOC)
*   **Communication:** ggwave, Bluetooth, Lora

## Roadmap & TODOs

Key areas for future development include:
*   Implementing high-speed vs. low-speed thinking models.
*   Developing a voice interrupt system.
*   Refining model selection and context management.
*   Implementing parallel TTS and ggwave transmission.
*   Exploring alternative technologies (e.g., faster-whisper, alternative TTS, simulation environments like Omniverse Isaac Sim, Dora-rs).
*   Integrating long-term memory and personality development.
*   CAD design for physical components (Head, BasePlate, Gimbal).
*   Finalizing hardware choices (Stereocams, Sensing Array, Comms).

*(This README is generated based on initial project notes and structure. It will be updated as the project evolves.)*
