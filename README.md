# Vector Embeddings Workshop – University of Vavuniya

This open-source project introduces students to vector embeddings, vector stores, and modern semantic search techniques. It was developed and delivered as a workshop for undergraduate students at the University of Vavuniya.

## 🔍 Overview
- Basics of vector representations
- Introduction to semantic similarity
- PCA-based dimensionality reduction
- 3D interactive visualization of embeddings
- Color mapping based on 3D vectors

## 📁 Project Structure
```
.
├── demo/                     # Source code and visual outputs
│   ├── Color vectors for demo.py
│   ├── colorful_vectors_3d.html
├── data/                     # Input embeddings (GloVe)
│   └── glove.6B.100d.txt
├── docs/                     # Presentation slides
│   └── Vector-Stores-and-Embeddings-The-Future-of-Search.pdf
├── notebooks/                # Jupyter demonstration notebook
│   └── embedding_demo.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

## 🚀 Getting Started
```bash
# Clone the repo
https://github.com/your-username/vector-embeddings-workshop-vavuniya.git

# Install dependencies
pip install -r requirements.txt

# Run the demo
python "demo/Color vectors for demo.py"

# Open the visualization
open "demo/colorful_vectors_3d.html"
```

## 📦 Requirements
- Python 3.7+
- numpy
- plotly
- scikit-learn
- matplotlib

## 🧠 Learnings
- Dimensionality reduction helps visualize high-dimensional embeddings
- Color mapping gives intuition about vector closeness
- Word embeddings (e.g. GloVe) can be visually explored in 3D

## 🌐 Live Interactive Demo

Explore the 3D visualization of word embeddings here:  
👉 [Interactive Plotly Demo](https://nisaharan.github.io/vector-embeddings-workshop-vavuniya/colorful_vectors_3d.html)

Rotate and hover to understand how words cluster in high-dimensional space!

![3D Vector Visualization Preview](assets/3d%20color%20vector%20visualization.png)

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgements
Thanks to students and faculty of the University of Vavuniya. Workshop and code developed by Nisaharan Genhatharan.