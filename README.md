# Arabic Quote Classifier - NLP Project 🤖📝

**Automatic Classification of Arabic Quotes using NLP Models**

## 🌟 Project Overview

This project implements an automatic classification system for Arabic quotes using Natural Language Processing (NLP) techniques. The system categorizes Arabic quotes into different thematic categories, addressing the unique linguistic and cultural challenges of Arabic text processing.

**Author**: Maryem Chakroun  
**Supervisor**: Mariem Gzara

## 📊 Dataset Statistics

- **Total Quotes**: 23,877 citations
- **Collection Method**: Web scraping from multiple Arabic sources
- **Data Sources**: Forums, blogs, social media platforms, news websites
- **Class Distribution**: Highly imbalanced dataset (Gini index: 0.9855)
  - Majority class: 905 instances
  - Minority class: 1 instance
  - Imbalance ratio: 905:1

### Category Examples
- **الكذب** (Lies)
- **الطموح والنجاح** (Ambition & Success)
- **التعاون** (Cooperation)
- **العلم والمعرفة** (Science & Knowledge)
- **الأمانة والنزاهة** (Integrity & Honesty)

## 🔧 Data Preprocessing Pipeline

### 1. **Diacritics Removal**
Removed Arabic diacritical marks to standardize text processing

### 2. **English Words Removal**
Eliminated English words that don't contribute to Arabic classification

### 3. **Insignificant Phrases Removal**
Removed repetitive and meaningless phrases to improve data quality

### 4. **Numbers Removal**
Cleaned numerical characters irrelevant to semantic classification

### 5. **Symbols & Special Characters Removal**
Removed special characters and symbols for text clarity

### 6. **Space & Non-Arabic Characters Cleanup**
Normalized consecutive spaces and removed non-Arabic characters

### 7. **Tokenization**
Split text into manageable tokens for better semantic analysis

### 8. **Stop Words Removal**
Removed common Arabic stop words (ال، من، في) using Arabic NLP libraries

### 9. **Data Sampling**
Applied sampling techniques to balance category distribution

### 10. **Light Stemming with Tashaphyne**
Used Tashaphyne library for Arabic stemming to reduce words to root forms

### 11. **Category Encoding**
Converted categorical labels to numerical values for machine learning

### 12. **TF-IDF Vectorization**
Transformed text to numerical features using TF-IDF weighting

## 🎯 Model Performance

### Chosen Model: **Decision Tree**

### Overall Results:
- **Log Loss (average)**: 3.7824
- **Macro F1 Score**: 0.5082
- **Weighted Accuracy**: 0.7438

### AUC Analysis:
- **92 classes**: AUC ≈ 0.5 (moderate performance)
- **2 classes**: AUC &lt; 0.5 (poor performance)
- **21 classes**: AUC = 1.00 (perfect performance)

### Additional Metrics:
- **Log Loss**: 2.9750
- **F1 Score**: 0.5241
- **Average Precision**: 0.7542

## 🏗️ Project Structure
arabic-quote-classifier/
│
├── data/
│   ├── raw/                    # 23,877 Arabic quotes
│   ├── processed/              # Preprocessed dataset
│   └── balanced/               # Sampled and balanced data
│
├── models/
│   ├── decision_tree.pkl       # Main classification model
│   └── preprocessing_pipeline.pkl
│
├── src/
│   ├── preprocessing/
│   │   ├── arabic_cleaner.py
│   │   ├── text_normalizer.py
│   │   └── tashaphyne_stemmer.py
│   │
│   ├── features/
│   │   └── tfidf_extractor.py
│   │
│   └── evaluation/
│       └── auc_analyzer.py
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── preprocessing_pipeline.ipynb
│   └── model_evaluation.ipynb
│
├── reports/
│   └── Rapport.docx            # Detailed project report
│
├── requirements.txt
├── README.md
└── main.py                     # Main training script


## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/maryemchk/arabic-quote-classifier.git
cd arabic-quote-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Arabic NLP resources
python -c "import nltk; nltk.download('stopwords')"
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/maryemchk/arabic-quote-classifier.git
cd arabic-quote-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Arabic NLP resources
python -c "import nltk; nltk.download('stopwords')"
```
###Training the Model
```bash
python main.py --model decision_tree --data data/processed/quotes_dataset.csv
```
## 📈 Performance Insights

### Strengths:
- **Perfect classification** for 21 categories (AUC = 1.00)
- **Good overall accuracy** (74.38%)
- **Effective preprocessing** pipeline for Arabic text

### Challenges:
- **Class imbalance** significantly affects performance
- **92 categories** show moderate performance requiring improvement
- **2 categories** with poor performance need special attention

### Recommendations:
1. **Data Augmentation**: Collect more samples for minority classes
2. **Advanced Sampling**: Implement SMOTE or other balancing techniques
3. **Feature Engineering**: Explore Arabic-specific linguistic features
4. **Ensemble Methods**: Combine multiple models for better performance
5. **Deep Learning**: Consider AraBERT or other Arabic language models

## 🔍 Technical Details

### Preprocessing Dependencies:
- **Tashaphyne**: Arabic light stemming
- **scikit-learn**: TF-IDF vectorization
- **NLTK**: Arabic stop words removal
- **Custom modules**: Arabic text normalization

### Model Configuration:
- **Algorithm**: Decision Tree Classifier
- **Features**: TF-IDF weighted word frequencies
- **Validation**: Cross-validation with AUC scoring
- **Evaluation**: Multi-class classification metrics

## 🤝 Contributing

We welcome contributions to improve the Arabic quote classification system!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📚 Future Work

- [ ] Implement data augmentation techniques
- [ ] Explore deep learning models (AraBERT, CAMeL Tools)
- [ ] Develop a web interface for real-time classification
- [ ] Create an API for mobile applications
- [ ] Expand dataset with more balanced sampling
- [ ] Implement ensemble learning approaches

## 📝 License

This project is part of academic research conducted under the supervision of Mariem Gzara.

## 📞 Contact

For questions or collaboration opportunities:
- **GitHub**: [@maryemchk](https://github.com/maryemchk)
