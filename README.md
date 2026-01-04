ClaimTrace: Media Forensic Dashboard
What is this?
I built VeritasLens to help figure out if news images and their captions are actually real or just AI-generated/edited. With so many deepfakes and fake news stories going around, I wanted to make a simple dashboard where you can upload a photo and get a "fact-check" report using a few different AI models.

How it works
The dashboard looks at a few things at once to see if an image is suspicious:

Error Level Analysis (ELA): Scans the image to see if any parts were photoshopped or tampered with.

AI Detector: Uses a Hugging Face model (umm-maybe) to see if the image was made by an AI.

Caption Checker: I used spaCy and distilroberta to compare the caption you give it against what the AI actually "sees" in the image. If they don't match, it flags it.

Visual Q&A: Uses BLIP-2 to describe the image and answer basic questions like "where was this taken?"

How the app is built
It's a pretty simple setup:

Backend: A FastAPI server that does all the heavy lifting and runs the AI models.

Frontend: Just basic HTML/CSS (Bootstrap) and JavaScript for the UI.

Tunneling: Since I'm running this in Colab, I used pyngrok so I can actually open the dashboard in a browser.

Preview
How to get it running
I designed this to run in Google Colab because those AI models are huge and need a GPU to work properly.

Upload Veritas_lens.ipynb to Colab.

Run the first three cells to install the libraries and load the models (this might take a few minutes).

Run the last cell to start the server.

Look for the Public URL in the output, click it, and the dashboard will open in a new tab.

(If you want to run it locally on your own PC, check the app.py and requirements.txt files, but you'll probably need a decent GPU!)
