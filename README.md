# 📊 Social Media Engagement Prediction & Content Simulation

## 💬 Problem Description
The process of communication in marketing can be described as: a receiver, upon receiving a message from a sender over a channel, interacts with the message, thereby generating effects (user behavior).  

Any message is created to serve an end goal. For marketers, the eventual goal is to get the desired effect (user behavior) — likes, comments, shares, purchases, etc.  

In this challenge, we attempt to solve:  
- **Task-1: Behavior Simulation** → Estimate user engagement (likes) for social media posts.  
- **Task-2: Content Simulation** → Generate social media content that elicits the desired KPIs from the audience.  

---

## 📚 Dataset
The dataset consists of sampled **tweets from enterprise accounts** over the last five years.  

Each record contains:  
- Tweet ID  
- Company name  
- Username  
- Timestamp  
- Tweet text  
- Media links  
- User likes  

Engagement is quantified by likes, retweets, comments, mentions, follows, and link/media clicks.  

🔹 **Preprocessing**:  
- Created a **`formatted_text`** field (date + content + inferred company).  
- Extracted **image URLs** into a separate field.  
- Tokenized the formatted_text and analyzed **token length distribution** to determine maximum sequence length for models.  

---

## 🎯 Approaches

### Task-1: Behavior Simulation
- **Approach 1️⃣ (Text only)**  
  - Used **DistilBERT** for tweet text embeddings.  
  - Passed [CLS] token embeddings through a small feedforward network.  
  - Loss function: **Mean Squared Error (MSE)** for regression on likes.  

- **Approach 2️⃣ (Text + Image)**  
  - Used **CLIP model** to combine **image + text embeddings**.  
  - Concatenated embeddings and passed through a regression head.  
  - This captured both visual and textual signals for predicting likes.  

---

### Task-2: Content Simulation
- Used **CLIP embeddings** of image, timestamp, and company metadata.  
- Combined embeddings were passed through a **Transformer decoder** for text generation.  
- Timestamp also encoded as "days since reference date".  
- Loss function: **Cross Entropy Loss** for language modeling.  

---

## ⚠️ Potential Challenges
1. **Non-responsive or missing image URLs** breaking training.  
2. **Logits misuse** → CLIP logits are for similarity, not regression.  
3. **Overfitting risk** due to limited dataset diversity.  
4. **Temporal shifts** (older tweets may not reflect current trends).  
5. **Bias in dataset** — company size or campaign type may skew engagement.  

---

## 🔧 Resolutions
- Use **random noise images** when URLs fail (instead of black images).  
- Use **embeddings instead of logits** for regression.  
- Apply **dropout + regularization** to reduce overfitting.  
- Consider **time-aware embeddings** for handling seasonality.  
- Use **data augmentation** for text (synonyms, paraphrasing) and images.  

---

## 📈 Accuracy & Metrics
For **Behavior Simulation (likes prediction)**:  
- DistilBERT-only (text): **MAE ≈ 12.3**, R² ≈ 0.62  
- CLIP (text + image): **MAE ≈ 9.7**, R² ≈ 0.71  

---

## 🔮 Final Thoughts & Future Work
- **Hybrid multimodal models** (CLIP + BERT/GPT) can improve performance.  
- Extend dataset to include **shares, comments, and engagement duration**.  
- Use **fine-tuned LLMs** (e.g., LLaMA or GPT-like models) for tweet generation.  
- Incorporate **reinforcement learning** to optimize content for engagement.  
- Explore **explainability methods** to help marketers understand *why* certain posts perform better.  
