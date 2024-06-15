# 🤖 Personal Assistant Chatbot

A modern, multi-session AI chatbot built with Streamlit, OpenAI, AWS SageMaker, and advanced RAG (Retrieval-Augmented Generation) techniques. Enjoy a beautiful, intuitive UI, secure authentication, document upload, and smart features like sentiment analysis, web search, and weather info—all in one assistant.

---

## ✨ Features

- **Modern UI**: Stylish chat bubbles, icons, and responsive sidebar for session management
- **Multi-Session Chat**: Save, rename, and manage multiple chat sessions with persistent memory
- **Document Upload**: Ingest PDF, DOCX, or TXT files for context-aware Q&A (RAG)
- **OpenAI & AWS SageMaker**: Combines LLMs and custom sentiment analysis for smarter responses
- **Web Search & Weather**: Ask for live web results or weather by location
- **Secure Auth**: Register/login with hashed passwords (DynamoDB)
- **Cloud Storage**: Sessions and files are securely stored in AWS S3
- **Customizable**: Easily adapt CSS and JS for your brand

---

## 🚀 Quickstart

### 1. **Clone the Repo**
```bash
git clone https://github.com/perlasaicharanreddy/chatbot.git
cd chatbot-main
```

### 2. **Install Requirements**
```bash
pip install -r requirements.txt
```

### 3. **Set Up Environment Variables**
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
Edit `.env` with your OpenAI, AWS, and API keys.

### 4. **Run the App**
```bash
streamlit run src/chatbot.py
```

---

## 🛠️ Configuration

- **OpenAI**: For LLM-powered chat and embeddings
- **AWS**: S3 for storage, DynamoDB for users, SageMaker for sentiment
- **SerpAPI**: For web search
- **OpenWeather**: For weather info

All keys are loaded from AWS Secrets Manager or `.env`.

---

## 💬 Usage Examples

### **Login/Register**
- Open the app in your browser
- Register a new account or log in with your credentials

### **Start Chatting**
- Type your question and hit **Ask**
- Upload a PDF, DOCX, or TXT to add context (RAG)
- The assistant will use your document to answer questions

### **Session Management**
- Use the sidebar to switch, rename, or delete chat sessions
- Click **New Chat** to start fresh

### **Advanced Features**
- **Web Search**: Ask questions like `search latest AI news`
- **Weather**: Ask `What's the weather?` (location permission required)
- **Sentiment Analysis**: The bot adapts its tone based on your mood

---

## 📦 Project Structure

```
chatbot-main/
├── src/
│   ├── chatbot.py        # Main Streamlit app
│   ├── css.py            # Custom CSS
│   ├── js.py             # Geolocation JS
├── requirements.txt      # Python dependencies
├── packages.txt          # System packages (if any)
├── logs/                 # Rotating logs
├── README.md             # This file
```

---

## 🧑‍💻 Sample Chat

```
User: Uploads a research.pdf
User: What are the main findings in this document?
Assistant: [Summarizes content from PDF]
User: search latest AI breakthroughs
Assistant: [Returns web search results]
User: What's the weather?
Assistant: The weather in your location is sunny with a temperature of 25°C.
```

---

## 🛡️ Security & Best Practices
- Passwords are hashed with bcrypt
- Sessions and files are stored in S3 (never local)
- All API keys are loaded securely
- Logs are rotated (daily, weekly, monthly)

---

## 🙌 Credits
- [Streamlit](https://streamlit.io/)
- [OpenAI](https://openai.com/)
- [AWS SageMaker](https://aws.amazon.com/sagemaker/)
- [LangChain](https://python.langchain.com/)
- [SerpAPI](https://serpapi.com/)
- [OpenWeather](https://openweathermap.org/)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
