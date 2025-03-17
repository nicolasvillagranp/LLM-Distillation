# **LLM Distillation**

## **Overview**
This project focuses on fine-tuning a **Large Language Model (LLM) for instruction following** under extreme hardware constraints (**Nvidia RTX 3060, 12GB VRAM**). Traditional model distillation approaches load both the **teacher and student models** onto the GPU simultaneously, comparing probability distributions at each forward pass. However, this method is highly inefficient in terms of memory and compute.

A more **efficient alternative** is to perform a **single forward pass** of the teacher model across the entire dataset, storing the resulting logits on disk. Since these logits remain constant throughout training, they can be accessed as needed without keeping the teacher model in memory. 

Storing the full logit distribution for each dataset instance is impractical due to memory constraints, but **research from ["BiLD: Bi-directional Logits Difference Loss for Large Language Model Distillation"](https://arxiv.org/abs/2406.13555) (Minchong Li, Feng Zhou, Xiaohui Song)** shows that **only a subset of the most significant logits is necessary** for effective distillation. This enables us to train a **larger student model** while **temporarily using a powerful Google Cloud machine at a significantly lower cost**. Additionally, memory requirements are minimized, as logits are accessed via a **dataloader**, eliminating the need for extensive RAM capacity.

---

## **Google Cloud Machine**
Due to hardware limitations, we leveraged **Google Cloud Compute Engine** to perform the teacher model's forward pass efficiently. 

**Machine Specifications:**
- **GPU:** Nvidia L4 (24GB VRAM)
- **CPU:** 8 vCPUs
- **Memory (RAM):** 32GB
- **Storage:** 100GB SSD
- **Provider:** Google Cloud Compute Engine  
- **Reasoning:** The L4 GPU was chosen for its balance between cost and performance, making it suitable for running inference on large-scale LLMs while keeping expenses manageable.

The Google Cloud VM was used to **generate and store the logits**, which were then used for student model training on a local machine.

---

## **Running the Code**
The project structure requires:
- A **`src/` folder** for code files.
- A **`data/` folder** (must exist before execution).

### **Execution Steps**
1. **Run `forward_big.py`**  
   - This script performs the **forward pass of the teacher model**.  
   - The generated logits are stored on disk.

2. **Run `train.py`**  
   - This script trains the **student model using the stored logits**.  
   - The resulting fine-tuned model is saved to disk.

---

## **Customization**
- The **LLM used can be changed** (this implementation is based on the **QWEN family**).
- Adjust the **batch size** for better training efficiency, depending on available hardware resources.

---

## **Future Work**
🔹 **Implement Gradient Checkpointing** – To further optimize memory usage during training.  
🔹 **Experiment with Different Logit Truncation Strategies** – To analyze the impact of varying the number of stored logits on model performance.  
🔹 **Optimize Student Model Architecture** – To find a more efficient trade-off between model size and performance.  
🔹 **Investigate Larger Teacher Models** – To assess whether an even larger teacher model would improve knowledge transfer.  
🔹 **Explore Mixed Precision Training** – To improve computational efficiency without sacrificing model accuracy.  

This approach allows efficient **fine-tuning of instruction-following models on constrained hardware**, leveraging **precomputed logits** to enable larger student models without requiring vast computational resources. 🚀
