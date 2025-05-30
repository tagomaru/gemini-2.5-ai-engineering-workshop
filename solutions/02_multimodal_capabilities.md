# Part 2 - Multimodal Capabilities (Images, Audio, Video, Documents)

Gemini models like `gemini-2.5-flash-preview-05-20` can process text, images, audio, video, and documents in a single prompt using `client.models.generate_content()`. This enables powerful multimodal AI applications that can understand and generate content across different media types.

**Key Capabilities:**
- **Visual Understanding**: Analyze images, extract text, identify objects
- **Audio Processing**: Transcribe speech, analyze music, understand audio content
- **Video Analysis**: Summarize videos, extract key frames, understand motion
- **Document Processing**: Extract information from PDFs, understand layouts
- **Multimodal Generation**: Create images and speech from text prompts

```python
from google import genai
from google.genai import types
import os
import sys
import requests
from PIL import Image
from io import BytesIO

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import userdata
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
else:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY',None)

# Create client with api key
MODEL_ID = "gemini-2.5-flash-preview-05-20"
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## 1. Image Understanding: Single Image

Gemini can analyze images in multiple formats: PIL `Image` objects, raw bytes, or uploaded files via the File API.

**When to use each method:**
- **PIL Images**: Small images (<20MB), local files, when you need image manipulation
- **Raw bytes**: When working with image data from APIs or memory
- **File API**: Large images (>20MB), when you want to reuse images across multiple requests

```bash
!curl -o image.jpg "https://storage.googleapis.com/generativeai-downloads/images/Cupcakes.jpg"
```

```python
from PIL import Image

# load image
image = Image.open("image.jpg")

response = client.models.generate_content(
    model=MODEL_ID,
    contents=["What is this image?", image])

print(response.text)
image
```

Or use an image as base64 encoded string.   

```python
image_url = "https://storage.googleapis.com/generativeai-downloads/images/scones.jpg"
image_response_req = requests.get(image_url)

prompt_specific = "Are there any fruits visible?"

response_specific = client.models.generate_content(
    model=MODEL_ID,
    contents=["What is this image?",
            types.Part.from_bytes(data=image_response_req.content, mime_type="image/jpeg")]
)
print(response_specific.text)
```

You can use the File API for large payloads (>20MB).

> The File API lets you store up to 20 GB of files per project, with a per-file maximum size of 2 GB. Files are stored for 48 hours. They can be accessed in that period with your API key, but cannot be downloaded from the API. It is available at no cost in all regions where the Gemini API is available.


```python
file_id = client.files.upload(file="../assets/data/Cupcakes.jpg")

response = client.models.generate_content(
    model=MODEL_ID,
    contents=["What is this image?", file_id]
)

print(response.text)
```

> The File API lets you store up to 20 GB of files per project, with a per-file maximum size of 2 GB. Files are stored for 48 hours. They can be accessed in that period with your API key, but cannot be downloaded from the API. It is available at no cost in all regions where the Gemini API is available.

## 2. Image Understanding: Multiple Images

Gemini can analyze and compare multiple images simultaneously, which is powerful for comparative analysis, visual storytelling, or understanding sequences.

```python
image_url_1 = "https://plus.unsplash.com/premium_photo-1694819488591-a43907d1c5cc?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8Y3V0ZSUyMGRvZ3xlbnwwfHwwfHx8MA%3D%3D" # Dog
image_url_2 = "https://images.pexels.com/photos/2071882/pexels-photo-2071882.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500" # Cat

image_response_req_1 = requests.get(image_url_1)
image_response_req_2 = requests.get(image_url_2)

pil_image1 = Image.open(BytesIO(image_response_req_1.content))
pil_image2 = Image.open(BytesIO(image_response_req_2.content))

prompt_multi_image = "Compare these two images. What are the main subjects in each, and what are they doing?"

response_multi = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        "Image 1:", pil_image1,
        "Image 2:", pil_image2,
        prompt_multi_image
    ]
)
print(response_multi.text)
```

## 3. !! Exercise: Product Description from Image !!

1.  **Image URL:** Find an image of a product (e.g., backpack, mug).
2.  **Prompt:** Ask model to identify, describe features, suggest use cases, and write a marketing slogan.
3.  **API Call:** Use `client.models.generate_content()` with the image (as PIL Image) and prompt.
4.  **Print Response**

```python
product_image_url = "https://images.unsplash.com/file-1705123271268-c3eaf6a79b21image?w=416&dpr=2&auto=format&fit=crop&q=60"


exercise_img_req = requests.get(product_image_url)
pil_image = Image.open(BytesIO(exercise_img_req.content))

exercise_prompt_text = """
    Based on the image provided:
    1. Identify the main product shown.
    2. Describe its key visual features (color, material if discernible, style).
    3. Suggest 2-3 potential use cases for this product.
    4. Write a short, catchy marketing slogan for it.
    """

exercise_response = client.models.generate_content(
    model=MODEL_ID,
    contents=[exercise_prompt_text, pil_image]
)
print(exercise_response.text)
```

#### **!! Additional Exercise !!**

Try analyzing different types of product images and compare the results:
- Fashion items (clothing, accessories)
- Electronics (gadgets, devices)
- Home goods (furniture, kitchenware)
- Outdoor equipment (sports, camping gear)

## 4. Audio Understanding

Gemini can process audio files for transcription, content analysis, speaker identification, and audio summarization. This is particularly useful for podcasts, meetings, interviews, and voice memos.

**Supported audio formats**: MP3, WAV, FLAC, AAC, and other common formats

```python
file_path = "../assets/data/audio.mp3"

file_id = client.files.upload(path=file_path)

# Generate a structured response using the Gemini API
prompt = """Generate a transcript of the episode. Include timestamps and identify speakers.

Speakers:
- John

eg:
[00:00] Brady: Hello there.
[00:02] Tim: Hi Brady.

It is important to include the correct speaker names. Use the names you identified earlier. If you really don't know the speaker's name, identify them with a letter of the alphabet, eg there may be an unknown speaker 'A' and another unknown speaker 'B'.

If there is music or a short jingle playing, signify like so:
[01:02] [MUSIC] or [01:02] [JINGLE]

If you can identify the name of the music or jingle playing then use that instead, eg:
[01:02] [Firework by Katy Perry] or [01:02] [The Sofa Shop jingle]

If there is some other sound playing try to identify the sound, eg:
[01:02] [Bell ringing]

Each individual caption should be quite short, a few short sentences at most.

Signify the end of the episode with [END].

Don't use any markdown formatting, like bolding or italics.

Only use characters from the English alphabet, unless you genuinely believe foreign characters are correct.

It is important that you use the correct words and spell everything correctly. Use the context of the podcast to help.
If the hosts discuss something like a movie, book or celebrity, make sure the movie, book, or celebrity name is spelled correctly."""
audio_part = genai.types.Part.from_uri(uri=audio_file.uri, mime_type=audio_file.mime_type)

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, audio_part]
)
print(response.text)
```


## 5. Video Understanding

Gemini can process video files to understand their content, analyze scenes, identify objects and actions, and provide detailed summaries.

**Video capabilities:**
- Scene analysis and summarization
- Object and action recognition
- Temporal understanding (what happens when)
- Content extraction and key moments
- YouTube video analysis

```python
video_path = "../assets/data/standup.mp4"

video_file_id = client.files.upload(path=video_path, display_name="Video 3 GenAI")

prompt = "Describe the main events in this video. What is the primary subject?"
video_part = genai.types.Part.from_uri(uri=video_file.uri, mime_type=video_file.mime_type)

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, video_part]
)

print(response.text)
```

### YouTube Video Analysis

The Gemini API supports direct YouTube URL analysis, which is very convenient for content analysis:

```python
# Analyze a YouTube video directly
youtube_url = "https://www.youtube.com/watch?v=dwgmfSOZNoQ"  # Google Cloud Next '25 Opening Keynote

youtube_part = genai.types.Part(
    file_data=genai.types.FileData(file_uri=youtube_url)
)
prompt = "What was the biggest Gemini announcement in this video?"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, youtube_part]
)

print(response.text)
```

## 6. !! Exercise: Summarize a YouTube Video !!

1.  **YouTube URL:** Find a YouTube video (e.g., a tutorial, news, or educational video).
2.  **Prompt:** Ask model to summarize the video or to generate a transcript.
3.  **API Call:** Use `client.models.generate_content()` with the YouTube URL and prompt.
4.  **Print Response**

```python
youtube_url = "https://www.youtube.com/watch?v=o7U4DV9Fkc0"

youtube_part = genai.types.Part(
    file_data=genai.types.FileData(file_uri=youtube_url)
)

prompt = "Summarize the video in 3 sentences."

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, youtube_part]
)

print(response.text)
```

**Try these variations:**
- Analyze a tutorial video and extract step-by-step instructions
- Summarize a news video and identify key facts vs. opinions
- Analyze a product review and extract pros/cons
- Process an educational video and create study notes

## 7. Working with PDF/Document Files

Gemini can extract information from PDFs and other document formats, making it excellent for document analysis, data extraction, and content summarization.

**Common use cases:**
- Invoice processing and data extraction
- Contract analysis and summarization
- Research paper analysis
- Form processing and validation
- Document classification and routing

```python
pdf_file_path = "../assets/data/rewe_invoice.pdf"

pdf_file_id = client.files.upload(path=pdf_file_path, display_name="Rewe Invoice")

prompt = "What is the total amount due?"
pdf_part = genai.types.Part.from_uri(uri=pdf_file_id, mime_type="application/pdf")

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, pdf_part]
)
print(response.text)
```

## 7. Code 

Gemini is good at understanding and generating code. Let's use [gitingest](https://github.com/cyclotruc/gitingest) to chat with a GitHub repo:

```
%pip install gitingest
```

```python
import gitingest

repo = gitingest.Repo("https://github.com/philschmid/nextjs-gemini-2-0-pdf-structured-data")

```

```python
print(summary)
```

```python
print(tree)
```

```python
prompt = f"""Explain what repository is about:

Code:
{content}
"""

chat = client.chats.create(model=MODEL_ID)

response = chat.send_message(prompt)
print(response.text)
```

```python
response = chat.send_message("How are the schemas defined?")
print(response.text)
```

```python
response = chat.send_message("Update all schema route to use the new Gemini 2.5 models, `gemini-2.5-flash-preview-05-20`. Return only the updated file.")
print(response.text)
```

## 9. Image Generation

Generate high-quality images using Gemini's image generation capabilities. This feature is perfect for creating visual content, prototypes, marketing materials, and creative projects.

**Image Generation Features:**
- Text-to-image generation
- Style control through prompts
- High-resolution output
- SynthID watermarking for authenticity
- Multiple aspect ratios and sizes

```python
from PIL import Image
from io import BytesIO


prompt_text = "A photo of a cat"

response = client.models.generate_content(
    model="gemini-2.0-flash-preview-image-generation",
    contents=prompt_text,
    config=types.GenerateContentConfig(
      response_modalities=['TEXT', 'IMAGE']
    )
)

# Process the response
image_saved = False
for part in response.candidates[0].content.parts:
  if part.text is not None:
    print(f"Text response: {part.text}")
  elif part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      image = Image.open(BytesIO(part.inline_data.data))
      image_filename = 'gemini_generated_image.png'
      image.save(image_filename)

image
```


**Image Generation Tips:**
- Be specific about style (photorealistic, illustration, cartoon, etc.)
- Include lighting and mood descriptors
- Specify composition details (close-up, wide shot, etc.)
- Mention art styles or references when relevant
- Consider aspect ratio and resolution needs

> **Note**: All generated images include a SynthID watermark for authenticity verification. More details in the [official documentation](https://ai.google.dev/gemini-api/docs/image-generation).

## 10. Text to Speech

Convert text into natural-sounding speech with controllable voice characteristics. This feature enables creating audio content, accessibility features, and interactive applications.

**TTS Capabilities:**
- Multiple voice options and styles
- Controllable pace, tone, and emotion
- Single-speaker and multi-speaker audio
- High-quality audio output
- Natural language voice direction

For this example, we'll use the `gemini-2.5-flash-preview-tts` model to generate single-speaker audio. You'll need to set the `response_modalities` to `["AUDIO"]` and provide a `SpeechConfig`.

```python
import soundfile as sf
import numpy as np
from IPython.display import Audio, display

text_to_speak = "Say cheerfully: Hello! Welcome to the world of generative AI audio."

response_tts = client.models.generate_content(
   model="gemini-2.5-flash-preview-tts", # Specific model for TTS
   contents=text_to_speak,
   config=types.GenerateContentConfig(
      response_modalities=["AUDIO"],
      speech_config=types.SpeechConfig(
         voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
               voice_name='Kore', # Choose from available prebuilt voices
            )
         )
      ),
   )
)

audio_array = np.frombuffer(response_tts.candidates[0].content.parts[0].inline_data.data, dtype=np.int16)
sf.write("generated_speech.wav", audio_array, 24000)
display(Audio(filename))
```

## !! Exercise: Avatar Generation !!

In this exercise, you will combine image generation and text-to-speech to create an avatar and have it introduce itself.

**Steps:**

1.  **Generate an Avatar Image:**
    *   Write a prompt to generate an image of a unique avatar (e.g., "A friendly, futuristic robot assistant with a welcoming smile, digital art style").
    *   Use the `gemini-2.0-flash-preview-image-generation` model to generate the image.
    *   Save and display the generated image.
2.  **Create an Introduction:**
    *   Write a short introductory sentence for your avatar (e.g., "Hello! I am your new AI assistant, ready to help you explore the wonders of technology.").
3.  **Generate Speech for the Introduction:**
    *   Use the `gemini-2.5-flash-preview-tts` model to convert the introduction text into speech.
    *   Choose a suitable voice (e.g., 'Puck' for an upbeat tone).
    *   Save the audio and provide a way to play it (e.g., using `IPython.display.Audio`).

**Bonus:** Try different prompts for the image and different voices or styles for the audio to see how the results change!

```python
prompt_avatar_image = "A friendly, futuristic robot assistant with a welcoming smile, digital art style, high resolution"

response_image = client.models.generate_content(
    model="gemini-2.0-flash-preview-image-generation",
    contents=prompt_avatar_image,
    config=types.GenerateContentConfig(
      response_modalities=['TEXT', 'IMAGE'] # TEXT is often included by default or good practice
    )
)


avatar_image = Image.open(BytesIO(response_image.candidates[0].content.parts[0].inline_data.data))
avatar_image.save("generated_avatar.png")

avatar_introduction_text = "Hello! I am Vision, your friendly AI assistant. I'm excited to help you generate amazing things!"

response_speech = client.models.generate_content(
   model="gemini-2.5-flash-preview-tts",
   contents=f"Say in an enthusiastic and clear voice: {avatar_introduction_text}",
   config=types.GenerateContentConfig(
      response_modalities=["AUDIO"],
      speech_config=types.SpeechConfig(
         voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
               voice_name='Puck', # An upbeat voice
            )
         )
      ),
   )
)

audio_array_speech = np.frombuffer(response_speech.candidates[0].content.parts[0].inline_data.data, dtype=np.int16)
sf.write("avatar_introduction.wav", audio_array_speech, 24000)
display(Audio("avatar_introduction.wav"))

display(avatar_image)
display(Audio("avatar_introduction.wav"))
```


## Recap & Next Steps

**What You've Learned:**
- Image understanding with single and multiple image analysis for various use cases
- Audio processing including speech transcription and audio content analysis
- Video analysis for scene understanding and YouTube content processing
- Document processing with PDF analysis and structured data extraction
- Code understanding for repository analysis and code review
- Creative generation with image creation and text-to-speech synthesis
- Multimodal integration combining different content types for rich applications
- File API usage for efficient handling of large files and reusable content

**Key Takeaways:**
- Use File API for large files (>20MB) and content you'll reuse multiple times
- Implement comprehensive error handling for network and API operations
- Structure prompts clearly and specifically for consistent, high-quality outputs
- Monitor token usage across different modalities for effective cost control
- Consider user experience and processing time for multimedia operations

**Next Steps:** Continue with [Part 3: Structured Outputs, Function Calling & Tools](https://github.com/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/solutions/03_structured_outputs_function_calling_tools.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/solutions/03_structured_outputs_function_calling_tools.ipynb)

**More Resources:**
- [Vision Understanding Documentation](https://ai.google.dev/gemini-api/docs/vision?lang=python)
- [Audio Understanding Documentation](https://ai.google.dev/gemini-api/docs/audio?lang=python)
- [Image Generation Guide](https://ai.google.dev/gemini-api/docs/image-generation)
- [Text-to-Speech Documentation](https://ai.google.dev/gemini-api/docs/speech-generation)