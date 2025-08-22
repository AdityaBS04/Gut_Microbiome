# 🧬 GutMLC Frontend Guide

## ✅ Setup Complete!

Your gut microbiome classification frontend is now ready to use!

### 🌐 Access the Web Interface

**The application is running at:**
- **Local**: http://127.0.0.1:5000
- **Network**: http://10.14.147.189:5000 (accessible from other devices on your network)

### 🚀 Quick Start

1. **Open your web browser** and navigate to: http://127.0.0.1:5000

2. **You'll see the GutMLC interface with two input options:**
   - **File Upload**: Drag & drop or click to upload CSV files
   - **Manual Input**: Enter 100 comma-separated abundance values

### 📊 How to Test

#### Option 1: Use Sample Data
1. Click the **"Use Sample Data"** button
2. This loads 100 random microbiome values into the manual input box
3. Click **"Predict from Manual Input"**

#### Option 2: Upload CSV File
1. Create a CSV file with microbiome data:
   ```csv
   sample_id,organism1,organism2,organism3,...
   sample1,0.1,2.3,1.5,...
   sample2,0.8,1.2,3.4,...
   ```
2. Drag the file to the upload area or click to browse
3. Get predictions automatically

#### Option 3: Manual Entry
1. Enter 100 comma-separated values like:
   ```
   0.1, 2.3, 1.5, 0.8, 1.2, 3.4, 0.5, 1.8, 2.1, 0.9, ...
   ```
2. Click **"Predict from Manual Input"**

### 📈 Understanding Results

The interface will show:
- **Top 5 disease predictions** for each sample
- **Confidence percentages** (probability scores)
- **Visual progress bars** for each prediction
- **Color-coded results** for easy interpretation

### 🔧 Current Status

**⚠️ Note**: The interface shows a warning that no trained model is loaded. This is because:
- The model training is still in progress (check quick mode: bash_1)
- You can still test the interface functionality with the sample data
- Once training completes, run: `python save_model_for_frontend.py` to enable full predictions

### 🎯 Supported Diseases (Current Test Set)

The frontend currently supports 10 diseases:
- COVID-19
- Anorexia  
- Hypertension
- Melanoma
- Tuberculosis
- Diabetes Mellitus
- Obesity
- Depression
- Asthma
- Arthritis, Rheumatoid

Once the full model is trained, all 89+ diseases will be supported.

### 🔥 Features

✅ **Drag & Drop File Upload**
✅ **Real-time Predictions** 
✅ **Sample Data Generator**
✅ **Beautiful, Responsive UI**
✅ **Top 5 Disease Rankings**
✅ **Progress Indicators**
✅ **Health Status Check**
✅ **Mobile-Friendly Design**

### 🛠️ Technical Details

- **Framework**: Flask + Bootstrap 5
- **ML Backend**: TensorFlow/Keras
- **Input**: 100-feature microbiome abundance vectors
- **Output**: Multi-label disease classification
- **Architecture**: 1D CNN with attention mechanism

### 📱 Screenshots

The interface includes:
- Modern gradient design with microbiome theme
- Interactive upload areas with hover effects
- Real-time prediction cards with progress bars
- Responsive layout that works on all devices

---

**🎉 Your GutMLC frontend is now live and ready for testing!**

Visit: **http://127.0.0.1:5000** to start using it!