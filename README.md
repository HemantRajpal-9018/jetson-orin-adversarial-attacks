# Jetson Orin Adversarial Attacks

Adversarial attack robustness testing on NVIDIA Jetson Orin Nano using PyTorch and ResNet50 for edge AI security research.

## 🎯 Project Overview

This project demonstrates adversarial attack vulnerability assessment on edge AI devices, specifically testing the NVIDIA Jetson Orin Nano's robustness against FGSM (Fast Gradient Sign Method) attacks. The research compares model predictions on AI-generated vs. real images and evaluates how adversarial perturbations affect inference on edge hardware.

## 🔬 Key Features

- **FGSM Targeted Adversarial Attacks**: Implementation of Fast Gradient Sign Method with targeted class manipulation
- **GPU-Accelerated Inference**: Optimized for NVIDIA Jetson Orin Nano edge computing
- **Dual Image Testing**: Comparative analysis between AI-generated and real images
- **Comprehensive Benchmarking**: Inference time, GPU memory usage, and prediction confidence tracking
- **JSON Result Export**: Structured output for analysis and visualization

## 📊 Results Summary

Based on experimental runs on Jetson Orin Nano:

### AI-Generated Image (dog_ai.webp)
- **Inference Time**: ~433ms
- **Clean Prediction**: Class 114 (29.47% confidence)
- **Adversarial Prediction**: Class 207 (27.74% confidence)
- **Attack Success**: ✅ Misclassified
- **Confidence Drop**: ~1.44%

### Real Image (dog.avif)
- **Inference Time**: ~31ms  
- **Clean Prediction**: Class 600 (1.95% confidence)
- **Adversarial Prediction**: Class 232 (48.31% confidence)
- **Attack Success**: ✅ Misclassified
- **Confidence Change**: -29.44% (increased)

## 🛠️ Tech Stack

- **Hardware**: NVIDIA Jetson Orin Nano Developer Kit
- **Framework**: PyTorch with CUDA acceleration
- **Model**: ResNet50 (pre-trained on ImageNet)
- **Attack Method**: FGSM (epsilon = 0.1)
- **Languages**: Python 3

## 📁 Repository Structure

```
jetson-orin-adversarial-attacks/
├── jetson3.py              # Main adversarial attack script
├── results/
│   └── gpu_adversarial_comparison.json  # Benchmark results
├── README.md
├── LICENSE
└── .gitignore
```

## 🚀 Usage

### Prerequisites
```bash
# Install dependencies
pip install torch torchvision pillow
```

### Running the Experiment
```python
python jetson3.py
```

The script will:
1. Load ResNet50 model on GPU
2. Process test images (AI-generated and real)
3. Perform clean inference
4. Apply FGSM adversarial attack
5. Compare predictions and save results to JSON

## 📈 Key Findings

1. **Both image types are vulnerable** to FGSM attacks with epsilon=0.1
2. **Real images show faster inference** (~31ms vs ~433ms for AI-generated)
3. **Adversarial perturbations successfully fool** the ResNet50 model on edge hardware
4. **Confidence changes vary significantly** between AI-generated and real images

## 🔐 Security Implications

This research highlights the importance of adversarial robustness in edge AI deployments, particularly for:
- **Autonomous systems** (robotics, drones)
- **Security cameras** and surveillance
- **Industrial IoT** applications
- **Mobile AI** applications

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{rajpal2024jetsonadversarial,
  author = {Hemant Rajpal},
  title = {Adversarial Attack Robustness Testing on NVIDIA Jetson Orin Nano},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/HemantRajpal-9018/jetson-orin-adversarial-attacks}
}
```

## 👤 Author

**Hemant Rajpal**
- AI/ML Engineer & Researcher
- Focus: Adversarial AI, Edge Computing, Cybersecurity
- LinkedIn: [Connect](https://linkedin.com/in/hemantrajpal)
- Email: hrajp2@unh.newhaven.edu

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA for Jetson Orin Nano hardware
- PyTorch team for the deep learning framework
- Research inspired by adversarial ML security community

---

**Note**: This research is for educational and security research purposes. Always ensure ethical use of adversarial AI techniques.
