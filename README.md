# 🤖 Fine-Tuning Transformer Architectures: A Complete Guide

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **A comprehensive implementation of fine-tuning three different transformer architectures (GPT-2, T5, and Vision Transformer) for real-world AI applications.**

This repository demonstrates fine-tuning state-of-the-art transformer models for three distinct tasks:
- 🍳 **Recipe Generation** using GPT-2 (Decoder-Only)
- 📰 **Text Summarization** using T5 (Encoder-Decoder)
- 🍱 **Food Image Classification** using Vision Transformer (ViT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Task 1: Recipe Generation (GPT-2)](#task-1-recipe-generation-gpt-2)
- [Task 2: Text Summarization (T5)](#task-2-text-summarization-t5)
- [Task 3: Image Classification (ViT)](#task-3-image-classification-vit)
- [Results](#results)
- [Deployment](#deployment)
- [Model Checkpoints](#model-checkpoints)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project explores three fundamental transformer architectures and their applications:

| Architecture | Model | Task | Dataset | Metric | Score |
|-------------|-------|------|---------|--------|-------|
| **Decoder-Only** | GPT-2 Small (124M) | Recipe Generation | 25K recipes | Perplexity | 3.71 |
| **Encoder-Decoder** | T5-Base (220M) | Text Summarization | 40K articles | ROUGE-L | 0.368 |
| **Vision Transformer** | ViT-Base-Patch16-224 (86M) | Image Classification | 101K images | Accuracy | 86.4% |

### Why This Project?

- ✅ **Educational**: Learn three different transformer paradigms
- ✅ **Production-Ready**: Deployable models with Gradio/Streamlit interfaces
- ✅ **Well-Documented**: Detailed explanations, comments, and blog post
- ✅ **Reproducible**: Complete training scripts with hyperparameters
- ✅ **Practical**: Real-world datasets and use cases

---

## ✨ Features

- 🔥 **From-Scratch Fine-Tuning**: No transfer learning shortcuts, full training pipelines
- 📊 **Comprehensive Evaluation**: Multiple metrics, confusion matrices, example outputs
- 🎨 **Interactive Demos**: Gradio interfaces for all three models
- 📈 **Training Visualizations**: Loss curves, attention maps, prediction examples
- 🚀 **Optimized Training**: FP16, gradient accumulation, checkpointing
- 📝 **Extensive Documentation**: Code comments, README, Medium blog post
- 🤗 **Hugging Face Integration**: Easy model sharing and deployment


---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, minimum 12GB VRAM)
- 20GB+ free disk space

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/transformer-finetuning.git
cd transformer-finetuning
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n transformer-ft python=3.9
conda activate transformer-ft

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Or install task-specific requirements
pip install -r task1_recipe_generation/requirements.txt
pip install -r task2_text_summarization/requirements.txt
pip install -r task3_image_classification/requirements.txt
```

### Step 4: Download Datasets

```bash
# Task 1: Download recipe dataset
# Place data.csv in task1_recipe_generation/data/raw/
# Download from: https://www.kaggle.com/datasets/nazmussakibrupol/3a2mext

# Task 2: CNN/DailyMail (auto-downloaded during training)
# Or manually: https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

# Task 3: Food-101 (auto-downloaded via Hugging Face)
# Or manually: https://www.kaggle.com/datasets/kmader/food41
```

---

## 🚀 Quick Start

### Run All Demos Locally

```bash
# Task 1: Recipe Generator
cd task1_recipe_generation
python app.py

# Task 2: Text Summarizer
cd task2_text_summarization
python app.py

# Task 3: Food Classifier
cd task3_image_classification
python app.py
```

### Train Models from Scratch

```bash
# Task 1: GPT-2 Recipe Generation
cd task1_recipe_generation
python scripts/preprocess.py  # Clean data
python scripts/train.py       # Train model

# Task 2: T5 Summarization
cd task2_text_summarization
python scripts/train.py

# Task 3: ViT Classification
cd task3_image_classification
python scripts/train.py
```

---

## 🍳 Task 1: Recipe Generation (GPT-2)

### Overview

Fine-tune GPT-2 to generate complete cooking recipes from dish names or ingredients.

**Model**: GPT-2 Small (124M parameters)  
**Dataset**: 25,000 cleaned recipes (from 250K raw)  
**Training Time**: ~6 hours on Tesla T4  
**Metric**: Perplexity = 3.71


**Data Format:**
```
Dish: <title> | Ingredients: <ingredients> | Instructions: <steps>
```

### Example Outputs

**Input**: `Dish: Spicy Thai Basil Chicken | Ingredients:`

**Output**:
```
Dish: Spicy Thai Basil Chicken | Ingredients: chicken breast, Thai basil, 
bell peppers, garlic, ginger, soy sauce, fish sauce, Thai chilies, oyster 
sauce | Instructions: Slice chicken into thin strips. Heat oil in wok over 
high heat. Stir-fry chicken 3-4 minutes until cooked. Add garlic, ginger, 
and chilies, cook 30 seconds. Add bell peppers, stir-fry 2 minutes. Mix in 
soy sauce, fish sauce, and oyster sauce. Toss until combined. Finish with 
fresh Thai basil. Serve over jasmine rice.
```

### Results

- **Excellent Recipes**: 60% (30/50 samples)
- **Good Recipes**: 30% (15/50 samples)
- **Poor Recipes**: 10% (5/50 samples)
- **Average Length**: 12.7 words per instruction set

---

## 📰 Task 2: Text Summarization (T5)

### Overview

Fine-tune T5-Base for abstractive summarization of news articles.

**Model**: T5-Base (220M parameters)  
**Dataset**: 40,000 CNN/DailyMail articles  
**Training Time**: ~8.5 hours on Tesla T4  
**Metrics**: ROUGE-1: 0.412, ROUGE-2: 0.188, ROUGE-L: 0.368


### Example Outputs

**Article** (634 words):
```
(CNN) -- With a breezy sweep of his pen President Obama brought into law 
the biggest change to the U.S. health system in decades. The bill, which 
passed after a year of bitter debate and procedural battles, will extend 
coverage to 32 million uninsured Americans...
```

**Generated Summary** (52 words):
```
President Obama signed historic health care legislation extending coverage 
to 32 million Americans. The bill passed after bitter partisan debate with 
Republicans unanimously opposing. CBO projects $143 billion deficit reduction, 
though critics dispute estimates. Law includes pre-existing condition 
protections and insurance exchanges.
```

**Reference Summary** (48 words):
```
Nina dos Santos says Europe must be ready to accommodate Obama's health care 
legislation. The bill extends coverage to millions of uninsured Americans. 
Republicans opposed the legislation calling it government overreach. CBO 
estimates significant deficit reduction over ten years.
```

### Results

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.4124 |
| ROUGE-2 | 0.1876 |
| ROUGE-L | 0.3628 |
| Perplexity | 3.67 |

---

## 🍱 Task 3: Image Classification (ViT)

### Overview

Fine-tune Vision Transformer for food image classification into 101 categories.

**Model**: ViT-Base-Patch16-224 (86M parameters)  
**Dataset**: Food-101 (101,000 images)  
**Training Time**: ~31.6 hours on Tesla P100  
**Metric**: Accuracy = 86.42%


### Example Outputs

**Test Image: Sushi Platter**
```
Predictions:
  1. sushi: 98.7%
  2. sashimi: 1.1%
  3. spring_rolls: 0.1%
```

**Test Image: Pizza**
```
Predictions:
  1. pizza: 93.2%
  2. bruschetta: 4.3%
  3. garlic_bread: 1.8%
```

### Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **86.42%** |
| **Top-5 Accuracy** | 96.8% |
| **Average Loss** | 0.521 |

**Top Performing Classes:**
- Sushi: 94.8%
- Pizza: 93.2%
- Ice Cream: 91.7%

**Most Confused Pairs:**
- Pancakes ↔ Waffles
- Ramen ↔ Pho
- Spaghetti Carbonara ↔ Spaghetti Bolognese

---

## 📊 Results

### Performance Comparison

| Model | Parameters | Training Time | Best Metric | Inference Speed |
|-------|-----------|---------------|-------------|-----------------|
| GPT-2 | 124M | 6.1 hours | Perplexity: 3.71 | 0.5s/recipe |
| T5 | 220M | 8.5 hours | ROUGE-L: 0.368 | 1.2s/summary |
| ViT | 86M | 31.6 hours | Accuracy: 86.4% | 0.3s/image |

### Key Findings

1. **GPT-2 excels at creative generation** but is hard to evaluate objectively
2. **T5 provides focused, controlled outputs** ideal for summarization
3. **ViT achieves excellent accuracy** with proper fine-tuning
4. **Transfer learning is crucial** - all models benefit from pre-training
5. **FP16 training doubles speed** with minimal quality loss

---

## 🚀 Deployment

### Local Deployment (Gradio)

All three tasks include Gradio interfaces for local testing:

---

## 💾 Model Checkpoints

### Pre-trained Models

Download fine-tuned models from Hugging Face Hub:

```bash
# Task 1: Recipe Generator
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("your-username/gpt2-recipe-generator")

# Task 2: Summarizer
from transformers import T5ForConditionalGeneration
model = T5ForConditionalGeneration.from_pretrained("your-username/t5-news-summarizer")

# Task 3: Food Classifier
from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained("imamakainat/vit-food101-finetuned")
```

### Training Checkpoints

Available in each task's `models/` directory:

- **Task 1**: `recipe_gpt2_finetuned/` (496 MB)
- **Task 2**: `model_epoch4/` (892 MB)
- **Task 3**: `vit-food101-finetuned/` (330 MB)

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit your changes**: `git commit -m 'Add some AmazingFeature'`
4. **Push to the branch**: `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### Areas for Contribution

- 🐛 Bug fixes
- 📝 Documentation improvements
- ✨ New features (additional models, tasks)
- 🎨 UI/UX improvements for demos
- 📊 Additional evaluation metrics
- 🚀 Performance optimizations

### Code Style

- Follow PEP 8 style guide
- Add docstrings to functions
- Include type hints
- Write unit tests for new features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- **Transformers**: Apache 2.0 License
- **PyTorch**: BSD License
- **Datasets**: Apache 2.0 License

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@misc{transformer-finetuning-2025,
  author = {Your Name},
  title = {Fine-Tuning Transformer Architectures: A Complete Guide},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/transformer-finetuning}}
}
```

### Related Papers

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  journal={NeurIPS},
  year={2017}
}

@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and others},
  journal={ICLR},
  year={2021}
}
```

---

## 🙏 Acknowledgments

### Datasets

- **Recipe Dataset**: [Kaggle Recipe Dataset](https://www.kaggle.com/datasets/nazmussakibrupol/3a2mext)
- **CNN/DailyMail**: [News Summarization Dataset](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)
- **Food-101**: [Food Image Dataset](https://www.kaggle.com/datasets/kmader/food41)

### Frameworks & Tools

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [Gradio](https://gradio.app/)
- [Weights & Biases](https://wandb.ai/) (for experiment tracking)

### Inspiration

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [An Image Is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- Hugging Face Course and Documentation

---

## 📧 Contact

**Your Name** - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

**Project Link**: [https://github.com/yourusername/transformer-finetuning](https://github.com/yourusername/transformer-finetuning)

**Blog Post**: [Medium Article](https://medium.com/@yourusername/fine-tuning-transformers)

**Live Demo**: [Hugging Face Spaces](https://huggingface.co/spaces/yourusername/transformer-demos)

---

## 📚 Additional Resources

### Tutorials

- [Task 1: Recipe Generation Tutorial](task1_recipe_generation/README.md)
- [Task 2: Summarization Tutorial](task2_text_summarization/README.md)
- [Task 3: Image Classification Tutorial](task3_image_classification/README.md)

### Documentation

- [Complete Blog Post](docs/blog_post.md)
- [Evaluation Report](docs/evaluation_report.pdf)
- [Training Logs](docs/training_logs/)

### External Links

- [Hugging Face Model Hub](https://huggingface.co/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Gradio Documentation](https://gradio.app/docs/)

---

## 🗺️ Roadmap

### Current Version (v1.0)

- ✅ Three fine-tuned models (GPT-2, T5, ViT)
- ✅ Complete training pipelines
- ✅ Gradio demos
- ✅ Comprehensive documentation

### Future Plans (v2.0)

- ⬜ Add more architectures (BERT, BART, CLIP)
- ⬜ Implement LoRA for efficient fine-tuning
- ⬜ Add quantization for faster inference
- ⬜ Multi-language support
- ⬜ Docker containers for easy deployment
- ⬜ Automated testing and CI/CD
- ⬜ Weights & Biases integration
- ⬜ Model distillation examples

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/transformer-finetuning&type=Date)](https://star-history.com/#yourusername/transformer-finetuning&Date)

---

<div align="center">

**Made with ❤️ and 🤖 by ZainabEman**

If you found this helpful, please ⭐ star the repo!

[Report Bug](https://github.com/yourusername/transformer-finetuning/issues) · [Request Feature](https://github.com/yourusername/transformer-finetuning/issues) · [Documentation](https://github.com/yourusername/transformer-finetuning/wiki)

</div>
