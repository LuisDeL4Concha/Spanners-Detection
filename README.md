# Spanners-Detection
 A custom RF-DETR v2 model for detecting spanners in images, optimized for edge deployment on resource-constrained devices. Delivering a trained model, configuration file. The model is tested on a custom spanner dataset. It is suitable for edge devices like Raspberry Pi or NVIDIA Jetson.

Features:
Custom RF-DETR v2 Model: Trainedfor spanner detection, with support for COCO-format datasets.
Edge Optimization: Input sizes optimized for edge devices (672x672), with performance benchmarking (837.9MB memory, 9.1 FPS).
Deployment Package: Includes model file (.pth), configuration file (.yml), and a standalone demo script.
Inference Demo: Processes test images, draws bounding boxes, and saves results for visualization.
Comprehensive Documentation: Configuration and deployment details in YAML and JSON formats.

Deliverables:
The project meets the Upwork client’s requirements:
Trained Model File: checkpoint_best_total.pth (pre-trained model; training code included for custom datasets).

Configuration File: deployment_config.yml with model and deployment details.

Results:
Images Processed: 10 test images from a custom spanner dataset.
Detections: 11 spanners detected with bounding boxes and confidence scores.

Performance:
Average Inference Time: 109.3ms
Average FPS: 9.1
Memory Usage: 837.9MB
Edge Compatibility Score: 100/100 (capped)

Setup Instructions:
Dependencies: pip install torch torchvision opencv-python numpy onnx onnxruntime psutil
Hardware: CPU (tested) or GPU (for improved performance); compatible with edge devices like Raspberry Pi or NVIDIA Jetson.
