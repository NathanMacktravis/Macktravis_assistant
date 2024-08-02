import base64
from threading import Lock, Thread
import cv2
import openai
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

load_dotenv()  # Load environment variables from a .env file


class WebcamStream:
    """Class for managing webcam stream in a separate thread."""
    
    def __init__(self):
        """Initialize the webcam stream and the first frame"""
        self.stream = VideoCapture(index=0) # To capture video from the webcam
        _, self.frame = self.stream.read() # Read the first frame
        self.running = False # Flag to indicate if the stream is running
        self.lock = Lock()  # To ensure thread-safe access to the webcam frame

     
    def start(self):
        """Function to start the webcam stream"""
        #Start the webcam stream in a separate thread if it's not already running
        if self.running:
            # Return the current instance if the stream is already running
            return self 
        
        self.running = True
        self.thread = Thread(target=self.update, args=()) # Create a new thread for the webcam stream
        self.thread.start() # Start the thread
        return self

    def update(self):
        """Function to continuously capture frames from the webcam"""
        # Continuously capture frames from the webcam
        while self.running:
            _, frame = self.stream.read() # Read the next frame
            self.lock.acquire() # Acquire the lock to update the frame
            self.frame = frame  # Update the latest frame
            self.lock.release() 

    def read(self, encode=False):
        """Function to read the current frame from the webcam"""
        # Read the current frame, optionally encoding it to base64
        self.lock.acquire()
        frame = self.frame.copy()  # Copy the frame to avoid modification during reading
        self.lock.release()

        if encode:
            _, buffer = imencode(".jpeg", frame) # Encode the frame as a JPEG image
            return base64.b64encode(buffer)  # Return the encoded frame

        return frame

    def stop(self):
        """Stop the webcam stream and the associated thread"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Release the webcam when exiting"""
        self.stream.release()  


class Assistant:
    """Class to handle interaction with the AI assistant model."""
    
    def __init__(self, model):
        """Initialize the assistant with a given model and create the inference chain"""
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        """Generate an answer based on the user's prompt and a webcam image"""
        if not prompt:
            return

        print("Prompt:", prompt)
        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip() # Invoke the inference chain and get the response
        print("Response:", response)

        if response:
            self._tts(response)  # Convert the response to speech

    def _tts(self, response):
        """Convert the AI's response to speech using OpenAI's TTS model"""
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response,
        ) as stream:
            try:
                for chunk in stream.iter_bytes(chunk_size=1024):
                    player.write(chunk)
            finally:
                player.close()  # Ensure the player is closed when done


    def _create_inference_chain(self, model):
        """Create a chain of tasks: processing input, model inference, and parsing output"""

        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.
        """
        
        # Define the prompt structure with a system message and placeholders
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        # Chain the tasks: processing the prompt, running the model, and parsing the response
        chain = prompt_template | model | StrOutputParser()

        # Store chat history for context
        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


# Initialize and start the webcam stream
webcam_stream = WebcamStream().start()

# Load a model for generating AI responses (Google's Gemini 1.5)
#model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# Optionally, we can switch to OpenAI's GPT-4 model instead
model = ChatOpenAI(model="gpt-4o", streaming=True)

# Create an assistant instance using the selected model
assistant = Assistant(model)


def audio_callback(recognizer, audio):
    """Handle audio input from the microphone, recognize speech, and generate a response"""
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, webcam_stream.read(encode=True))  # Pass the prompt and webcam image to the assistant

    except UnknownValueError:
        print("There was an error processing the audio.")  # Handle speech recognition errors


# Initialize speech recognition and set up the microphone
recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise

# Start listening to the microphone in the background and handle incoming audio
stop_listening = recognizer.listen_in_background(microphone, audio_callback)

# Continuously display the webcam feed and wait for a key press to exit
while True:
    cv2.imshow("webcam", webcam_stream.read())
    if cv2.waitKey(1) in [27, ord("q")]:  # Exit on 'ESC' or 'q' key press
        break

# Stop the webcam stream and cleanup resources
webcam_stream.stop()
cv2.destroyAllWindows()
# Stop listening to the microphone in the background
stop_listening(wait_for_stop=False)  
