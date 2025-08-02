# Deep Learning Projects

This repository contains various deep learning experiments and projects using TensorFlow, Keras, and PyTorch.

## 📚 Project Overview

This repository serves as a collection of deep learning implementations, tutorials, and experiments. Currently featuring comprehensive MNIST digit classification analysis with overfitting/underfitting comparisons.

## 🏗️ Project Structure

```
Deep-Learning/
├── Lab 2/
│   ├── mnist.ipynb              # Main MNIST analysis notebook
│   ├── mnist_analysis.html      # Exported HTML report
│   └── mnist.pdf               # PDF export (if available)
├── requirements.txt            # Python dependencies
├── venv/                      # Virtual environment
└── README.md                  # This file
```

## 🔬 Lab 2: MNIST Digit Classification Analysis

### Overview
Comprehensive analysis of neural network performance on the MNIST dataset, demonstrating:
- **Underfitting**: Too simple model architecture
- **Balanced Model**: Optimal architecture for the task
- **Overfitting**: Too complex model architecture

### Key Features
- ✅ Complete model comparison with visualizations
- ✅ Training vs validation accuracy/loss plots
- ✅ Overfitting analysis and recommendations
- ✅ Performance metrics and gap analysis
- ✅ Best practices for model selection

### Model Architectures Tested

1. **Underfitting Model** (Too Simple)
   ```python
   model = Sequential([
       Flatten(input_shape=(28, 28)),
       Dense(10, activation='softmax')
   ])
   ```

2. **Balanced Model** ⭐ (Recommended)
   ```python
   model = Sequential([
       Flatten(input_shape=(28, 28)),
       Dense(128, activation='relu'),
       Dense(10, activation='softmax')
   ])
   ```

3. **Overfitting Model** (Too Complex)
   ```python
   model = Sequential([
       Flatten(input_shape=(28, 28)),
       Dense(512, activation='relu'),
       Dense(512, activation='relu'),
       Dense(256, activation='relu'),
       Dense(128, activation='relu'),
       Dense(10, activation='softmax')
   ])
   ```

### Results Summary
- **Balanced Model Performance**: ~98% accuracy with minimal overfitting
- **Training Time**: ~50 epochs for comprehensive analysis
- **Dataset**: MNIST (60k training, 10k test images)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/harshajustin/Deep-Learning.git
   cd Deep-Learning
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebooks

1. **Start Jupyter Lab**
   ```bash
   jupyter lab
   ```
   
2. **Open the desired notebook**
   - Navigate to `Lab 2/mnist.ipynb` for MNIST analysis

3. **Run all cells** to see the complete analysis

## 📦 Dependencies

### Core Deep Learning Frameworks
- **TensorFlow 2.20.0rc0** - Main deep learning framework
- **PyTorch 2.7.1** - Alternative deep learning framework
- **Keras 3.11.0** - High-level neural networks API

### Data Science & Visualization
- **NumPy 2.2.6** - Numerical computing
- **Pandas 2.3.1** - Data manipulation
- **Matplotlib 3.10.3** - Plotting and visualization
- **Seaborn 0.13.2** - Statistical data visualization

### Machine Learning
- **Scikit-learn 1.7.1** - Machine learning utilities
- **SciPy 1.16.1** - Scientific computing

### Development Environment
- **Jupyter Lab 4.4.5** - Interactive development
- **IPython 9.4.0** - Enhanced Python shell

*See `requirements.txt` for complete dependency list*

## 📊 Key Visualizations

The MNIST analysis includes:
- **Training/Validation Curves**: Track model performance over epochs
- **Accuracy Comparison Charts**: Side-by-side model performance
- **Overfitting Analysis**: Gap analysis between training and validation
- **Model Architecture Diagrams**: Visual representation of network structures

## 🎯 Learning Objectives

### Completed ✅
- [x] Understanding overfitting vs underfitting
- [x] Model architecture design principles
- [x] Performance visualization and analysis
- [x] Best practices for neural network training
- [x] MNIST dataset handling and preprocessing

### Future Labs 🔮
- [ ] Convolutional Neural Networks (CNNs)
- [ ] Transfer Learning
- [ ] Recurrent Neural Networks (RNNs)
- [ ] Generative Adversarial Networks (GANs)
- [ ] Computer Vision projects
- [ ] Natural Language Processing

## 📈 Results & Insights

### Key Findings from MNIST Analysis
1. **Optimal Architecture**: 128 hidden neurons provide best balance
2. **Generalization**: Simple architecture generalizes better than complex ones
3. **Training Efficiency**: Balanced model trains faster and more reliably
4. **Overfitting Prevention**: Monitoring validation metrics is crucial

### Performance Metrics
| Model Type | Training Acc | Validation Acc | Gap | Status |
|------------|-------------|----------------|-----|---------|
| Underfitting | ~85% | ~85% | ~0% | Poor Performance |
| **Balanced** | **~98%** | **~97%** | **~1%** | **Optimal** ✅ |
| Overfitting | ~99% | ~92% | ~7% | Poor Generalization |

## 🛠️ Tools & Technologies

- **Development**: VS Code, Jupyter Lab
- **Version Control**: Git, GitHub
- **Documentation**: Markdown, HTML export
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Environment**: Python virtual environments

## 📝 Export & Sharing

### Available Formats
- **Jupyter Notebook** (`.ipynb`) - Interactive format
- **HTML Export** (`.html`) - Shareable web format
- **PDF Export** (`.pdf`) - Print-ready format

### Export Commands
```bash
# Export to HTML
jupyter nbconvert --to html Lab\ 2/mnist.ipynb

# Export to PDF (requires LaTeX)
jupyter nbconvert --to pdf Lab\ 2/mnist.ipynb
```

## 🤝 Contributing

Feel free to contribute to this project by:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

**Harsha Justin**
- GitHub: [@harshajustin](https://github.com/harshajustin)
- Project Link: [https://github.com/harshajustin/Deep-Learning](https://github.com/harshajustin/Deep-Learning)

## 🙏 Acknowledgments

- MNIST dataset from Yann LeCun's website
- TensorFlow and Keras documentation
- Deep Learning community tutorials and best practices
- VS Code and Jupyter development environments

---

⭐ **Star this repository if you found it helpful!**

*Last updated: August 2025*
