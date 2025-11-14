import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time
import json
import os

# Create results directory
os.makedirs("results", exist_ok=True)

# ═══════════════════════════════════
# TITLE: Create results directory
# ═══════════════════════════════════

device = torch.device("cuda")
print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")

# ═══════════════════════════════════
# TITLE: Force GPU
# ═══════════════════════════════════

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device).eval()
print(f"✓ Model loaded on GPU")

# ═══════════════════════════════════
# TITLE: Load model on GPU
# ═══════════════════════════════════

images = {
    "ai_generated": "/home/raiden/Downloads/dog_ai.webp",
    "real": "/home/raiden/Downloads/dog.avif"
}

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

results_all = {}

for img_type, img_path in images.items():
    print("=" * 50)
    print(f"Testing: {img_type.upper()}")
    print("=" * 50)
    
    # ═══════════════════════════════════
    # TITLE: Test both images
    # ═══════════════════════════════════
    
    img = Image.open(img_path)
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    print(f"✓ Loaded: {img_path}")
    
    # ═══════════════════════════════════
    # TITLE: Load image
    # ═══════════════════════════════════
    
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        output = model(img_tensor)
    torch.cuda.synchronize()
    inference_time = (time.time() - start) * 1000
    print(f"✓ Inference time: {inference_time:.2f}ms")
    
    # ═══════════════════════════════════
    # TITLE: Benchmark inference
    # ═══════════════════════════════════
    
    pred_class = torch.argmax(output, dim=1).item()
    clean_conf = torch.softmax(output, dim=1).max().item()
    print(f"✓ Clean prediction confidence: {clean_conf * 100:.2f}%")
    
    # ═══════════════════════════════════
    # TITLE: Get prediction
    # ═══════════════════════════════════
    
    epsilon = 0.1
    img_tensor_attack = img_tensor.clone().requires_grad_(True)
    
    # ═══════════════════════════════════
    # TITLE: FGSM Attack (targeted to fool the model)
    # ═══════════════════════════════════
    
    target_class = (pred_class + 1) % 1000  # Different class
    output_attack = model(img_tensor_attack)
    loss = torch.nn.functional.cross_entropy(output_attack, torch.tensor([target_class]).to(device))
    loss.backward()
    
    adv_img = img_tensor_attack - epsilon * img_tensor_attack.grad.sign()
    adv_img = torch.clamp(adv_img, -2, 2)
    
    # ═══════════════════════════════════
    # TITLE: Target a DIFFERENT class to fool the model
    # ═══════════════════════════════════
    
    with torch.no_grad():
        adv_output = model(adv_img)
    
    adv_class = torch.argmax(adv_output, dim=1).item()
    adv_conf = torch.softmax(adv_output, dim=1).max().item()
    
    # ═══════════════════════════════════
    # TITLE: Test adversarial
    # ═══════════════════════════════════
    
    misclassified = (adv_class != pred_class)
    confidence_drop = (clean_conf - adv_conf) * 100
    
    print(f"✓ Original prediction: Class {pred_class} ({clean_conf * 100:.2f}%)")
    print(f"✓ Adversarial prediction: Class {adv_class} ({adv_conf * 100:.2f}%)")
    print(f"✓ Misclassified: {misclassified}")
    print(f"✓ Confidence change: {confidence_drop:.2f}%")
    
    # ═══════════════════════════════════
    # TITLE: Check if prediction changed
    # ═══════════════════════════════════
    
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    print(f"✓ GPU Memory used: {gpu_mem:.2f}GB")
    
    # ═══════════════════════════════════
    # TITLE: GPU Memory
    # ═══════════════════════════════════
    
    results_all[img_type] = {
        "inference_time_ms": round(inference_time, 2),
        "clean_confidence_percent": round(clean_conf * 100, 2),
        "clean_class": pred_class,
        "adversarial_class": adv_class,
        "adversarial_confidence_percent": round(adv_conf * 100, 2),
        "misclassified": misclassified,
        "confidence_change_percent": round(confidence_drop, 2),
        "gpu_memory_gb": round(gpu_mem, 2)
    }
    
    # ═══════════════════════════════════
    # TITLE: Store results
    # ═══════════════════════════════════

final_results = {
    "device": "GPU (Orin)",
    "model": "ResNet50",
    "attack_method": "FGSM (targeted to different class)",
    "epsilon": 0.1,
    "images": results_all
}

with open("results/gpu_adversarial_comparison.json", "w") as f:
    json.dump(final_results, f, indent=2)

print("=" * 50)
print("✓ Results saved to results/gpu_adversarial_comparison.json")
print("=" * 50)

# ═══════════════════════════════════
# TITLE: Save all results
# ═══════════════════════════════════

print("\n📊 COMPARISON SUMMARY:")
print(f"✓ AI-generated misclassified: {results_all['ai_generated']['misclassified']}")
print(f"✓ Real dog misclassified: {results_all['real']['misclassified']}")
print(f"✓ AI-generated inference: {results_all['ai_generated']['inference_time_ms']:.2f}ms")
print(f"✓ Real dog inference: {results_all['real']['inference_time_ms']:.2f}ms")

# ═══════════════════════════════════
# TITLE: Summary comparison
# ═══════════════════════════════════
