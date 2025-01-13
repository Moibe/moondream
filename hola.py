# Install dependencies in your project directory
# pip install moondream

import moondream as md
from PIL import Image
import bridges

# Initialize with API key
model = md.vl(api_key=bridges.moondream)

# Load an image
image = Image.open("chimpanzees--community-mother.webp")
encoded_image = model.encode_image(image)  # Encode image (recommended for multiple operations)


# Generate a caption (length options: "short" or "normal" (default))
caption = model.caption(encoded_image)["caption"]
print("Caption:", caption)

# Stream the caption
for chunk in model.caption(encoded_image, stream=True)["caption"]:
    print(chunk, end="", flush=True)

# Ask a question
answer = model.query(encoded_image, "What's in this image?")["answer"]
print("Answer:", answer)

# Stream the answer
for chunk in model.query(encoded_image, "What's in this image?", stream=True)["answer"]:
    print(chunk, end="", flush=True)

# Detect objects
detect_result = model.detect(image, 'subject')  # change 'subject' to what you want to detect
print("Detected objects:", detect_result["objects"])

# Point at an object
point_result = model.point(image, 'subject')  # change 'subject' to what you want to point at
print("Points:", point_result["points"])