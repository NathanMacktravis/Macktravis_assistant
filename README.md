# Macktravis_assistant
This application integrates AI models to provide interactive responses using a webcam and speech recognition. It can handle user queries by analyzing images from the webcam and processing voice commands.
It very slow with two models, but it works ! 😅 

## Features

- **Webcam Stream**: Captures and processes live video feed.
- **Speech Recognition**: Converts spoken commands into text using OpenAI's Whisper model.
- **AI Responses**: Uses Google Gemini 1.5 (or optionally OpenAI GPT-4) to generate responses based on text prompts and images.
- **Text-to-Speech**: Converts AI responses into spoken words.

## Technologies Used

- **OpenAI**: For generating text responses and text-to-speech.
- **Google Gemini**: For AI model responses.
- **LangChain**: For managing chat history and integrating AI models.
- **OpenCV**: For webcam stream handling.
- **Speech Recognition**: For processing audio input.
- **PyAudio**: For audio output.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NathanMacktravis/Macktravis_assitant
   cd Macktravis_assitant
   ```

2. **Create a virtual environment and activate it**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the root directory of the project and add the following keys with your respective API keys:

   ```env
   OPENAI_API_KEY=<your-openai-api-key>
   GOOGLE_API_KEY=<your-google-api-key>
   ```

## Usage

1. **Run the application**:
   ```bash
   python app.py
   ```

2. **Interact with the assistant**:
   - Speak commands or ask questions to the microphone.
   - The assistant will respond based on the AI model's output and the webcam feed.

3. **Stop the application**:
   - Press `ESC` or `q` while the webcam window is focused.

## Notes

- Ensure your webcam and microphone are properly connected and accessible.
- Modify the `app.py` file to switch between different AI models as needed (Google Gemini or OpenAI GPT-4), I marked where you need to modify in comment 😉. 
