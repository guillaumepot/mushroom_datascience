# 🍄 Mushroom Classification using Deep Learning

<img src="./media/mushroom_img.jpeg" width="350" height="350" alt="Mushroom Classification Project">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

This project implements a comprehensive deep learning pipeline for mushroom species classification using computer vision techniques. Originally developed as a data science diploma project, it has evolved into a full-featured application with data preprocessing, model training, evaluation, and an interactive web interface.

### Key Features

- **🔍 Data Analysis**: Comprehensive exploratory data analysis of mushroom observation datasets
- **🖼️ Image Processing**: Advanced image preprocessing and augmentation pipeline
- **🧠 Deep Learning**: Transfer learning with EfficientNetV2B0 for accurate classification
- **📊 Visualization**: Interactive data visualization and model evaluation dashboards
- **🌐 Web Interface**: User-friendly Streamlit application for exploration and inference
- **📈 Experiment Tracking**: MLflow integration for model versioning and tracking

## 🏗️ Architecture

```
mushroom_datascience/
├── src/
│   ├── notebooks/          # Jupyter notebooks for the ML pipeline
│   │   ├── 1-dataset_clean.ipynb       # Data cleaning and preprocessing
│   │   ├── 2-data_analysis.ipynb       # Exploratory data analysis
│   │   ├── 3-classification_choice.ipynb # Classification strategy selection
│   │   ├── 4-file_sort.ipynb           # Image file organization
│   │   ├── 5-imgs_analysis.ipynb       # Image analysis and statistics
│   │   ├── 6-data_preprocessing.ipynb  # Dataset preparation for ML
│   │   ├── 7-modelization.ipynb        # Model architecture design
│   │   ├── 8-model_train.ipynb         # Model training pipeline
│   │   └── 9-model_evaluate.ipynb      # Model evaluation and metrics
│   │
│   ├── streamlit/          # Web application interface
│   │   ├── main.py                     # Main Streamlit application
│   │   ├── home_page.py               # Homepage with project overview
│   │   ├── datasetcsv_page.py         # Dataset exploration interface
│   │   ├── dataanalysis_page.py       # Data analysis dashboard
│   │   ├── imageviz_page.py           # Image visualization tools
│   │   ├── modelization_page.py       # Model architecture display
│   │   ├── evaluation_page.py         # Model performance metrics
│   │   └── streamlit_vars.py          # Shared variables and configurations
│   │
│   ├── utils/              # Utility functions and helpers
│   │   ├── file_processing.py          # File operations and management
│   │   ├── img_preprocessing.py        # Image processing functions
│   │   ├── import_df.py               # Data import utilities
│   │   └── model_generation.py        # Model architecture functions
│   │
│   └── logs/               # Application logs
│
├── storage/                # Data storage directory
│   ├── datas/
│   │   ├── csv/                       # CSV datasets
│   │   ├── imgs/                      # Image datasets
│   │   ├── json/                      # Metadata and configurations
│   │   └── tf_datasets/               # TensorFlow datasets
│   │
│   ├── models/             # Trained models and artifacts
│   │   ├── builded/                   # Model architectures
│   │   ├── trained/                   # Trained model weights
│   │   ├── histories/                 # Training histories
│   │   └── model_checkpoint/          # Training checkpoints
│   │
│   └── mlruns/             # MLflow experiment tracking
│
├── config/                 # Configuration files
├── old/                   # Legacy project files and reports
├── media/                 # Project media and assets
├── pyproject.toml         # Poetry dependency management
├── requirements.txt       # Pip dependencies
└── README.md             # This file
```

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.10 or higher
- **Storage**: At least 2GB free space for datasets and models
- **Memory**: Recommended 8GB+ RAM for model training

### Installation

#### Option 1: Using Poetry (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd mushroom_datascience

# Install dependencies with Poetry
poetry install
poetry shell
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd mushroom_datascience

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies

```yaml
Python: "^3.10"
TensorFlow: "^2.16.1"
Streamlit: "^1.32.2"
Pandas: "^2.2.1"
Scikit-learn: "^1.4.1"
OpenCV: "^4.9.0"
MLflow: "^2.12.1"
Seaborn: "^0.13.2"
```

## 📊 Dataset

The project uses a comprehensive mushroom observation dataset containing:

- **647,623 initial observations** (after cleaning: ~600K)
- **33 features** including taxonomic information, images, and metadata
- **Image dataset** with cleaned mushroom photographs
- **GBIF taxonomic data** for species classification

### Data Pipeline

1. **Data Cleaning** (`1-dataset_clean.ipynb`): Remove duplicates, handle missing values
2. **Analysis** (`2-data_analysis.ipynb`): Exploratory data analysis and visualization
3. **Image Processing** (`4-file_sort.ipynb`, `5-imgs_analysis.ipynb`): Image organization and analysis
4. **Preprocessing** (`6-data_preprocessing.ipynb`): Feature engineering and dataset preparation

## 🤖 Machine Learning Pipeline

### Model Architecture

The project implements a transfer learning approach using **EfficientNetV2B0** as the backbone:

```python
# Model Structure
Base Model: EfficientNetV2B0 (ImageNet weights)
├── Global Average Pooling
├── Dense Layer (1024 units)
├── Dropout (0.3)
├── Dense Layer (512 units)
├── Dropout (0.3)
├── Dense Layer (256 units)
└── Output Layer (softmax activation)
```

### Key Features

- **Transfer Learning**: Leverages pre-trained EfficientNetV2B0
- **Data Augmentation**: Comprehensive image augmentation pipeline
- **Batch Processing**: Optimized data loading with TensorFlow datasets
- **Experiment Tracking**: MLflow integration for reproducible experiments

### Training Pipeline

1. **Model Design** (`7-modelization.ipynb`): Architecture definition
2. **Training** (`8-model_train.ipynb`): Model training with callbacks
3. **Evaluation** (`9-model_evaluate.ipynb`): Performance assessment and metrics

## 🌐 Web Application

Launch the interactive Streamlit application to explore the project:

```bash
cd src/streamlit
streamlit run main.py
```

### Application Features

- **🏠 Home**: Project overview and navigation
- **📄 Dataset CSV**: Interactive dataset exploration
- **📈 Data Analysis**: Comprehensive data visualization dashboard
- **🖼️ Image Visualization**: Image dataset exploration tools
- **🔧 Modelization**: Model architecture visualization
- **📊 Evaluation**: Model performance metrics and analysis

## 🛠️ Usage

### Quick Start

1. **Setup Environment**:
   ```bash
   poetry install && poetry shell
   ```

2. **Prepare Data**:
   ```bash
   # Run data preprocessing notebook
   jupyter notebook src/notebooks/6-data_preprocessing.ipynb
   ```

3. **Train Model** (optional):
   ```bash
   # Run model training notebook
   jupyter notebook src/notebooks/8-model_train.ipynb
   ```

4. **Launch Application**:
   ```bash
   cd src/streamlit
   streamlit run main.py
   ```

### Advanced Usage

#### Custom Model Training

```python
from src.utils.model_generation import generate_model
from src.utils.img_preprocessing import generate_dataset

# Generate custom model
model = generate_model(num_classes=10, input_shape=(224, 224, 3))

# Create datasets with custom preprocessing
train_ds, val_ds, test_ds = generate_dataset(
    target_to_encode=labels,
    features_to_split=features,
    augment=True,
    normalize=True
)
```

#### Batch Image Processing

```python
from src.utils.img_preprocessing import img_preprocessing

# Process single image
processed_img = img_preprocessing(
    img_url="path/to/image.jpg",
    resize=True,
    normalize=True,
    augment=True
)
```

## 📈 Results

The model achieves competitive performance on mushroom classification:

- **Training Accuracy**: ~85-90%
- **Validation Accuracy**: ~80-85%
- **Model Size**: ~50MB (optimized for deployment)
- **Inference Time**: <100ms per image

*Note: Detailed metrics available in the evaluation notebook and Streamlit dashboard.*

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `src/streamlit/` directory:

```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Model Configuration

Adjust model parameters in `src/utils/model_generation.py`:

```python
# Model hyperparameters
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 10  # Adjust based on your dataset
DROPOUT_RATE = 0.3
DENSE_UNITS = [1024, 512, 256]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Guillaume Pot**
- 📧 Email: guillaume.pot@outlook.com
- 💼 LinkedIn: [Connect with me](https://linkedin.com/in/guillaume-pot)

## 🙏 Acknowledgments

- Data science diploma program that inspired this project
- GBIF for providing comprehensive taxonomic data
- TensorFlow and Streamlit communities for excellent documentation
- EfficientNet authors for the pre-trained model architecture

## 🔄 Project Status

This project is actively maintained and open for contributions. It serves as both a learning resource and a practical application of modern machine learning techniques in computer vision.

---

**Note**: This is a research/educational project not intended for production use without proper validation and testing.