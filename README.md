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

![Chatbot Screenshot](Screenshot%202025-05-26%20at%203.48.32%E2%80%AFPM.png)

---

## 🚀 Quickstart

> **Note:** You can run the chatbot locally (see below) or deploy it to the cloud using Docker and AWS EC2 (see [☁️ Deploying on AWS EC2 with Docker](#%EF%B8%8F-deploying-on-aws-ec2-with-docker)).

### 1. Clone the Repo
```zsh
git clone https://github.com/perlasaicharanreddy/chatbot.git
cd chatbot-main
```

### 2. Install Requirements
```zsh
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
```zsh
cp .env.example .env
# Edit .env with your OpenAI, AWS, and API keys
```

### 4. Run the App Locally
```zsh
streamlit run src/chatbot.py
```

---

## ☁️ Deploying on AWS EC2 with Docker

A step-by-step guide to deploy your chatbot on AWS EC2 using Docker.

### 1. IAM Role Setup
- Go to **AWS Console → IAM → Roles → Create role**
- Select **EC2** as trusted entity
- Attach policies: `AmazonEC2FullAccess`, `CloudWatchLogsFullAccess`
- Name and create the role (e.g., `EC2ChatbotRole`)

### 2. Launch EC2 Instance
- Use **Amazon Linux 2023** AMI
- Choose instance type (e.g., `t2.micro`)
- Create/download a PEM file (e.g., `chatbot.pem`)
- Allow inbound ports: `22` (SSH), `80` (HTTP)
- Attach the IAM role
- Launch the instance

### 3. Connect to EC2
```zsh
chmod 600 chatbot.pem
ssh -i chatbot.pem ec2-user@<EC2_PUBLIC_IP>
```

### 4. Install Docker
```zsh
sudo yum update -y
sudo yum -y install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
# Log out and back in for group changes to take effect
```

### 5. Transfer Files
```zsh
scp -i chatbot.pem Dockerfile requirements.txt docker-compose.yml -r src/ ec2-user@<EC2_PUBLIC_IP>:/home/ec2-user/downloads
```

### 6. Build & Run Docker Container
```zsh
cd /home/ec2-user/downloads
sudo docker build -t chatbot:v1.0 -f Dockerfile .
sudo docker run -t -p 80:8080 chatbot:v1.0
```

### 7. Access the App
- Open `http://<EC2_PUBLIC_IP>` in your browser

### 8. Useful Docker Commands
- List containers: `docker ps`
- Stop a container: `docker stop <container_id>`
- Build new image: `docker build -t chatbot:v2.0 -f Dockerfile .`

### 9. Security Best Practices
- Keep your `.pem` file secure (`chmod 600 chatbot.pem`)
- Restrict security group rules to only necessary ports
- Use IAM roles for EC2 (avoid hardcoded AWS credentials)
- Regularly update your EC2 instance and Docker images
- Terminate EC2 when not in use to avoid charges

---

## ⚙️ Configuration

- **OpenAI**: For LLM-powered chat and embeddings
- **AWS**: S3 for storage, DynamoDB for users, SageMaker for sentiment
- **SerpAPI**: For web search
- **OpenWeather**: For weather info

All keys are loaded from AWS Secrets Manager or `.env`.

---

## 💬 Usage Guide

### Authentication
- Open the app in your browser
- Register a new account or log in

### Chatting & Document Upload
- Type your question and hit **Ask**
- Upload a PDF, DOCX, or TXT to add context (RAG)
- The assistant will use your document to answer questions

### Session Management
- Use the sidebar to switch, rename, or delete chat sessions
- Click **New Chat** to start fresh

### Advanced Features
- **Web Search**: Ask `search latest AI news`
- **Weather**: Ask `What's the weather?` (location permission required)
- **Sentiment Analysis**: The bot adapts its tone based on your mood

---

## 🗂️ Project Structure

```
chatbot-main/
├── src/
│   ├── chatbot.py        # Main Streamlit app
│   ├── css.py            # Custom CSS
│   ├── js.py             # Geolocation JS
├── Dockerfile            # Docker build instructions
├── requirements.txt      # Python dependencies
├── packages.txt          # System packages
├── chatbot.pem           # SSH key for EC2 (keep secure, not in repo)
├── logs/
│   ├── chatbot_daily.log
│   ├── chatbot_weekly.log
│   └── chatbot_monthly.log
├── README.md             # Project documentation
├── LICENSE               # License file
├── Screenshot 2025-05-26 at 3.48.32 PM.png # Example screenshot
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
- Use IAM roles and security groups for cloud deployments
- Keep your PEM and credentials private

---

## 🙌 Credits

- [Streamlit](https://streamlit.io/)
- [OpenAI](https://openai.com/)
- [AWS](https://aws.amazon.com/)
- [Amazon EC2](https://aws.amazon.com/ec2/)
- [AWS SageMaker](https://aws.amazon.com/sagemaker/)
- [Docker](https://www.docker.com/)
- [LangChain](https://python.langchain.com/)
- [SerpAPI](https://serpapi.com/)
- [OpenWeather](https://openweathermap.org/)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
