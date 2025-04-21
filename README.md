# whatsapp-chat-analysis
A streamlit app to analyze your whatsapp chats

Demo Link: https://whatsapp-chat-analyzer-luvjain3.streamlit.app/
Image : 

# WhatsApp Chat Analyzer

![WhatsApp Chat Analyzer](https://img.shields.io/badge/WhatsApp-Chat%20Analyzer-25D366?style=for-the-badge&logo=whatsapp&logoColor=white)

A powerful tool that provides detailed insights and visualizations from your WhatsApp conversations. Discover patterns, statistics, and metrics about your chats with an interactive and user-friendly interface.

![WhatsApp Chat Analyzer Screenshot](https://i.imgur.com/lVuOIbG.jpg)

## 📊 Features

### Overview Analysis
- Total message count, word count, media shared, and links shared
- Most active users in group chats
- Monthly and daily chat timeline visualization

### Activity Analysis
- Daily and hourly activity tracking
- Week and month activity patterns
- Activity heatmap by day and hour

### Content Analysis
- Word cloud generation from chat content
- Most common words analysis
- Emoji usage analysis
- Message length distribution

### Sentiment Analysis
- Message sentiment distribution (positive, neutral, negative)
- Sentiment trends over time
- User sentiment comparison

### Advanced Metrics
- Response time analysis between messages
- Chat activity patterns and momentum
- Chat initiative analysis (who starts conversations)
- Media sharing patterns
- User engagement metrics comparison

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/whatsapp-chat-analyzer.git
cd whatsapp-chat-analyzer
```

Create a virtual environment (optional but recommended):
```bash
python -m venv venv
```

Activate the virtual environment:
- On Windows:
  ```bash
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

Install the required packages:
```bash
pip install -r requirements.txt
```

### Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## 📱 How to Export Your WhatsApp Chat

1. Open the WhatsApp chat you want to analyze
2. Tap the three dots (menu) in the top right
3. Select "More" > "Export chat"
4. Choose "Without media"
5. Save the exported .txt file
6. Upload the file to the WhatsApp Chat Analyzer

## 🛠️ Technologies Used

- **Streamlit**: For creating the interactive web application
- **Pandas**: For data manipulation and analysis
- **NLTK**: For sentiment analysis
- **Matplotlib & Seaborn**: For data visualization
- **Plotly**: For interactive charts
- **WordCloud**: For generating word clouds
- **URLExtract**: For extracting URLs from messages

## ✨ Features in Detail

### Theme Customization
Choose from multiple themes to personalize your analysis experience:
- WhatsApp Green
- Dark Mode
- Ocean Blue
- Sunset Orange

### Export Options
Export your analysis data in various formats:
- CSV
- Excel
- JSON

### Interactive Visualizations
All charts and graphs are interactive, allowing you to:
- Hover for detailed information
- Zoom in/out
- Pan across visualizations
- Download charts as images

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgements

- Created by Luv Jain, Neelesh Pahuja, and Pratham Shukla
- Built with ❤️ for WhatsApp chat enthusiasts

---

Made with 💬 by WhatsApp Chat Analyzer Team
