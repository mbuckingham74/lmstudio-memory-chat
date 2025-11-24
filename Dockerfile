FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your scripts
COPY . .

# Expose Gradio port
EXPOSE 7860

# Run the Gradio app
CMD ["python", "chat_ui_gradio.py"]
