---
title: ALU Chatbot
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
license: mit
---

# ALU Chatbot

This is the backend for the ALU Student Companion Chatbot, providing information about African Leadership University through a conversational interface.

## Features
- Question answering about ALU policies and procedures
- Enhanced capabilities including math solving and web search
- Conversation memory for context-aware responses

# ALU Student Companion Chatbot

A sophisticated, AI-powered conversational assistant designed specifically for African Leadership University students. This chatbot provides comprehensive information about academic programs, policies, campus life, and more while maintaining natural conversational abilities.

![ALU Chatbot](https://img.shields.io/badge/ALU-Student%20Companion-blue)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)
![Render](https://img.shields.io/badge/Render-Deployed-green)

## 📋 Table of Contents

- Features
- Technology Stack
- Architecture
- Installation
- Development Setup
- Deployment
- API Documentation
- Project Structure
- Contributing
- License

## ✨ Features

- **ALU Knowledge Base**: Comprehensive information about ALU academic policies, programs, admissions, campus life, and more
- **Conversational AI**: Natural language understanding using Hugging Face's distilGPT2 model
- **Document Retrieval**: Semantic search across university documentation with vector embeddings
- **Enhanced Capabilities**:
  - Web search for general knowledge questions
  - Mathematical problem solving
  - Code understanding and explanation
  - Context-aware responses based on conversation history
- **Multi-platform Deployment**: Accessible via Hugging Face Spaces and web frontend

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.10+
- **NLP & AI**:
  - Hugging Face Transformers (distilGPT2, CLIP)
  - Sentence Transformers for embeddings
  - ChromaDB for vector storage
- **Data Processing**:
  - Document processing (PDF, DOCX)
  - Structured knowledge JSON files
- **Deployment**:
  - Hugging Face Spaces
  - Render

## 🏗️ Architecture

The system uses a multi-capability architecture:

1. **Message Classification**: Determines if a query is ALU-related or general
2. **Capability Router**: Routes queries to specialized handlers:
   - Greeting/farewell detection
   - Document search for ALU information
   - Web search for general knowledge
   - Math solver for equations
   - Code support for programming questions
3. **Conversational Fallback**: Handles general conversation when specialized capabilities don't apply
4. **Response Formatting**: Creates well-structured, helpful responses

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/alu-student-companion.git
cd alu-student-companion/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Development Setup

```bash
# Start the FastAPI server locally
uvicorn main:app --reload --port 8000

# The API will be available at:
# http://localhost:8000

# API documentation will be available at:
# http://localhost:8000/docs
```

## 🌐 Deployment

### Hugging Face Spaces

This chatbot is deployed on Hugging Face Spaces:
- URL: [https://ngum-alu-chatbot.hf.space](https://ngum-alu-chatbot.hf.space)

To deploy your own instance:

1. Create a Hugging Face account
2. Create a new Space with the FastAPI template
3. Push your code to the Space repository:

```bash
git remote add space https://huggingface.co/spaces/your-username/your-space-name
git push space main
```

### Render

The frontend application is deployed on Render:
- URL: [https://alu-student-companion.onrender.com](https://alu-student-companion.onrender.com)

## 📚 API Documentation

The backend API provides the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Main chatbot interaction endpoint |
| `/api/alu-events` | GET | Get upcoming events at ALU |
| `/api/reset-conversation` | POST | Reset conversation history |
| `/api/health` | GET | Health check endpoint |

For detailed API documentation, visit `/docs` when running the server.

## 📁 Project Structure

```
backend/
├── app.py                  # FastAPI application entry point
├── main.py                 # Core chatbot logic and API routes
├── requirements.txt        # Python dependencies
├── enhanced_capabilities/  # Specialized chatbot capabilities
│   ├── capability_router.py # Routes questions to appropriate handlers
│   ├── conversation_model.py # Conversational AI model
│   ├── math_solver.py      # Handles mathematical questions
│   ├── web_lookup.py       # Web search for general knowledge
│   └── code_support.py     # Code understanding capabilities
├── data/                   # Knowledge base and document storage
│   ├── academic_policies.json
│   ├── academic_programs.json
│   └── ...
├── prompt_engine/          # Response generation components
├── retrieval_engine.py     # Document retrieval and vector search
└── vector_index/           # Vector embeddings for document search
```

## 👥 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -am 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Submit a pull request

For major changes, please open an issue first to discuss what you'd like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Developed for African Leadership University students. For support, please contact [your-contact-info].