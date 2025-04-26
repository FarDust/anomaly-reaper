# 🌌 Anomaly Reaper: Astronomical Image Anomaly Detection

**Detect anomalies in astronomical images using embeddings and PCA**

![Project Version](https://img.shields.io/badge/version-0.1.0-blue)

## 🎯 Overview

Anomaly Reaper is a specialized tool for detecting unusual patterns and anomalies in astronomical imagery data. By leveraging embedding models and Principal Component Analysis (PCA), it can identify outlier patterns that may represent interesting astronomical phenomena or artifacts in TESS (Transiting Exoplanet Survey Satellite) imagery.

## ✨ Key Features

- 🖼️ Process FITS astronomical image files from TESS missions
- 🔄 Generate embeddings from image data using Google Vertex AI
- 📊 Reduce dimensionality with PCA and detect outliers
- 🚀 API interface for uploading and processing new images
- 📊 Interactive visualization of detected anomalies
- 🔍 Search and filter anomalies by characteristics

## 🛠️ Technology Stack

- **Backend:** Python, FastAPI, SQLAlchemy
- **Machine Learning:** Google Vertex AI embeddings, scikit-learn
- **Image Processing:** Astropy, scikit-image, PIL
- **Visualization:** Matplotlib, Plotly, Seaborn
- **Data Management:** SQLite

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or later
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Install dependencies with UV:
   ```bash
   # Install all dependencies including dev and test groups
   uv --group dev --group test sync
   ```

2. Run the application:
   ```bash
   anomaly-reaper run
   ```

The API will be available at http://localhost:8000.

> **Note:** The project uses Google Application Default Credentials for authentication. Make sure you have them configured on your system. You can set them up by running `gcloud auth application-default login` if you have the Google Cloud SDK installed.

## 📊 How It Works

1. **Image Processing:**
   - Imports FITS files (Flexible Image Transport System)
   - Extracts meaningful patches around detected objects
   - Applies normalization, denoising, and contrast enhancement

2. **Embedding Generation:**
   - Uses Google Vertex AI's multimodal embedding model to extract 1408-dimensional embeddings
   - Captures semantic features of astronomical objects

3. **Anomaly Detection:**
   - Applies PCA to reduce embedding dimensionality
   - Calculates reconstruction error for each image
   - Flags images with high reconstruction error as potential anomalies

4. **Visualization:**
   - Projects data onto principal components for visualization
   - Highlights anomalous samples for exploration

## 🔧 Configuration

Configuration can be updated using the CLI:

```bash
# Configure anomaly detection threshold
anomaly-reaper configure --threshold 2.5

# Set custom directories
anomaly-reaper configure --models-dir ./models --uploads-dir ./uploads

# Configure database
anomaly-reaper configure --db-url sqlite:///custom.db
```

## 📋 Development Timeline

| Phase | Task | Status |
|-------|------|--------|
| 1️⃣ | Environment & Package Setup | ✅ |
| 2️⃣ | FITS Image Loading & Processing | ✅ |
| 3️⃣ | Embedding Generation via Google Vertex AI | ✅ |
| 4️⃣ | PCA & Anomaly Detection Implementation | ✅ |
| 5️⃣ | API Development | 🔄 |
| 6️⃣ | Visualization & Dashboard | 🔄 |

## 📊 Example Output

When running the anomaly detection process, you'll see:

1. Processed astronomical images
2. Embeddings visualization via PCA
3. Highlighted anomalies based on reconstruction error
4. Sorted list of most anomalous objects for investigation

## 💡 Use Cases

- Discovering unusual stellar objects in astronomical surveys
- Identifying instrument artifacts in image data
- Spotting potential transient phenomena in time-series imagery
- Automating the anomaly review process for large datasets

## 🔜 Future Development

- Improved clustering methods for better anomaly categorization
- Additional embedding model options
- More sophisticated anomaly detection algorithms
- Advanced time-series analysis for variable phenomena

## 📄 License

This project is licensed under the MIT License.

---

Created for astronomical image analysis and anomaly detection.