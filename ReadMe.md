# 🚨 ForensAI - AI-Powered Crime Scene Analysis System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**ForensAI** is a comprehensive AI-powered crime scene analysis system that combines Natural Language Processing, Computer Vision, and Machine Learning to assist law enforcement agencies in crime scene investigation and evidence analysis.

## 🎯 Key Features

### 🔍 **Multi-Modal Analysis**
- **Text Analysis**: FIR classification and key information extraction
- **Visual Evidence**: Object detection in crime scene images
- **Timeline Reconstruction**: Automatic event sequencing from reports
- **Cross-Modal Validation**: Text and visual evidence correlation

### 🚨 **Crime Scene Intelligence**
- **Real-time Object Detection** using YOLOS transformer model
- **Weapon and Evidence Recognition** with confidence scoring
- **Suspicious Pattern Detection** and risk assessment
- **Historical Case Matching** for investigation insights

### 📊 **Investigation Support**
- **Risk Assessment Framework** with urgency factors
- **Timeline Visualization** of reconstructed events
- **Evidence Documentation** with confidence warnings
- **Actionable Recommendations** for investigators

### 💻 **Production Features**
- **Interactive Web Interface** built with Streamlit
- **Detection Result Caching** for improved performance
- **Export Functionality** (TXT reports, JSON data)
- **Real-time Analysis** with progress tracking

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
CUDA (optional, for GPU acceleration)
```

### Installation
```bash
# Clone the repository
git clone https://github.com/shrehs/ForensAI.git
cd ForensAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app/streamlit_app.py
```

### Usage
1. **Upload FIR Text**: Paste crime scene description
2. **Add Timeline Events**: Specify key timestamps
3. **Upload Images**: Crime scene photographs
4. **Analyze**: Get comprehensive AI analysis
5. **Export Results**: Download reports and data

## 🏗️ System Architecture

```
ForensAI/
├── 📁 app/                    # Streamlit web interface
├── 📁 agent/                  # AI reasoning agents
├── 📁 nlp/                    # Natural language processing
├── 📁 vision/                 # Computer vision modules
├── 📁 data/                   # Historical cases and cache
├── 📁 models/                 # Pre-trained AI models
├── scene_pipeline.py          # Core analysis pipeline
└── requirements.txt           # Python dependencies
```

## 🔧 Core Components

### **Scene Pipeline**
- Orchestrates multi-modal analysis
- Coordinates AI agents and models
- Generates comprehensive reports

### **Object Detection**
- YOLOS transformer for scene analysis
- Crime-relevant object filtering
- Confidence-based validation

### **Timeline Extraction**
- Pattern-based event sequencing
- Temporal relationship analysis
- Evidence correlation

### **Risk Assessment**
- Multi-factor risk evaluation
- Urgency level classification
- Protective measure recommendations

## 📈 Performance Features

### **Caching System**
- Detection result caching
- Model weight persistence
- Faster re-analysis

### **Quality Assurance**
- Confidence scoring for all predictions
- False-positive filtering
- Manual verification prompts

### **Scalability**
- Batch processing support
- GPU acceleration
- Modular architecture

## 🎯 Use Cases

### **Law Enforcement**
- Initial crime scene assessment
- Evidence cataloging and analysis
- Investigation prioritization
- Officer safety evaluation

### **Legal System**
- Case preparation assistance
- Evidence organization
- Pattern recognition
- Documentation standardization

### **Training & Education**
- Investigation technique training
- Case study analysis
- Forensic education support

## 🔬 Technical Specifications

### **AI Models**
- **Object Detection**: YOLOS (You Only Look Once at Scenes)
- **Text Classification**: Pattern-based intent recognition
- **Timeline Analysis**: Rule-based event extraction
- **Risk Assessment**: Multi-factor evaluation framework

### **Accuracy Metrics**
- Object Detection: 75%+ confidence threshold
- Crime Classification: Pattern-based validation
- Timeline Reconstruction: Confidence scoring
- Risk Assessment: Multi-factor analysis

## 📊 Sample Analysis Output

```
🎯 ANALYSIS SUMMARY:
Crime Type: Domestic Violence (87% confidence)
Risk Level: 🔴 HIGH RISK
Key Evidence: Weapon (baseball bat), witness testimony
Timeline: 3 reconstructed events
Recommendations: Immediate victim protection, collect bat for forensics
```

## 🛡️ Ethical Considerations

- **Human Oversight Required**: All AI decisions need human validation
- **Transparency**: Clear confidence scores and reasoning provided
- **Bias Mitigation**: Diverse training data and validation processes
- **Privacy Protection**: Secure handling of sensitive case data

## 🚧 Future Enhancements

- **Advanced NLP**: Integration with large language models
- **Video Analysis**: Motion detection and behavioral analysis
- **Mobile App**: Field investigation support
- **Database Integration**: Case management system connectivity, a better corpus too
- **Multi-language Support**: International deployment capability

## 🤝 Contributing

We welcome contributions!

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black . && flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/shrehs/ForensAI/issues)
- **Email**: [shreyahs2004@gmail.com]

## 🙏 Acknowledgments

- YOLOS model by Hugging Face
- Streamlit for the web framework
- PyTorch and Transformers libraries
- Open source crime analysis research community

---

**⚠️ Disclaimer**: ForensAI is designed to assist human investigators and should not be used as the sole basis for legal decisions. All AI-generated insights require human validation and verification.

**NOTE: The project is still in progress**: 

**🌟 Star this repository if you find it useful!**