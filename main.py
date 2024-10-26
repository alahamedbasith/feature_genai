import os
import cv2
import pandas as pd
import numpy as np
import json
import io
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from PIL import Image
from tensorflow.keras.models import load_model

# Load environment variables from the .env file
load_dotenv()
# Set up the Gemini API using the key stored in the environment
import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API"))

# Load the cattle registration data from a CSV file
df_cattle = pd.read_csv('D:/Cattle Identification/sample_registration.csv')

# Load the class labels from a JSON file
label_json_path = 'D:/Cattle Identification/class_labels_vgg.json'  
with open(label_json_path, 'r') as json_file:
    labels = json.load(json_file)

# Load the pre-trained VGG model
model = load_model('D:/Cattle Identification/vgg_model.h5')

# Initialize the FastAPI application
app = FastAPI()

# Function to prepare the image for the model
def load_and_preprocess_image(img):
    # Resize the image to 224x224 pixels
    img = cv2.resize(img, (224, 224))  
    img_array = np.array(img)  # Convert the image to a NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    img_array = img_array.astype('float32') / 255.0  # Normalize the pixel values to [0, 1]
    return img_array

# Function to predict the cattle class based on the image
def predict_cattle_class(model, img_array, threshold=0.55):
    predictions = model.predict(img_array)  # Get predictions from the model
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Find the class index with the highest probability
    predicted_class_prob = float(predictions[0][predicted_class_index])  # Get the probability of the predicted class

    # Check if the predicted probability exceeds the threshold
    if predicted_class_prob > threshold:
        predicted_class = labels[str(predicted_class_index)]  # Map the index to the class label using the JSON
        return predicted_class, predicted_class_prob
    else:
        return None, predicted_class_prob  # Return None if the probability is below the threshold

# Function to retrieve registration details based on predicted class
def display_registration_details(predicted_class):
    if predicted_class:
        registration_details = df_cattle[df_cattle['Class'] == predicted_class]  # Get details for the predicted class
        if not registration_details.empty:
            details_list = []  # Create a list to hold registration details
            for _, row in registration_details.iterrows():
                details = {
                    "Cattle ID": row['Cattle ID'],
                    "Cattle Breed": row['Breed'],
                    "Cattle Age Average": row['Age (Years)'],
                    "Owner Name": row['Owner Name'],
                    "Owner Contact": row['Owner Contact'],
                    "Registration Date": row['Registration Date']
                }
                details_list.append(details)  # Add the details to the list
            return details_list  # Return the list of details as JSON
        else:
            return []  # Return an empty list if no details are found
    else:
        return []  # Return an empty list if predicted_class is None 

# Function to interact with Gemini's API and get a response
def get_gemini_response(input_text, image):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, image])  # Get a response from the model
    print("Gemini Response:", response.text)  # Log the response for debugging
    return response.text

# Root endpoint that displays a form for image upload
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Cattle Management</title>
        </head>
        <body>
            <h1>Cattle Management</h1>
            <form action="/analyze" method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/jpeg, image/png" required>
                <button type="submit">Tell me about the image</button>
            </form>
        </body>
    </html>
    """

# Combined prompt for image classification
combined_prompt = (
    "Classify the uploaded image based on the following criteria: "
    "1. If the image contains a cow's muzzle and the muzzle is large enough to indicate a close-up (muzzle covers at least 15% of the image), return 'yes'. Otherwise, return 'no'. "
    "2. Focus specifically on the muzzle area, including the nose and mouth. The area must be clearly visible, in focus, and close-up to be classified as 'yes'. "
    "3. If the image is far from the camera, shows a farm or landscape, or does not have a clear cow muzzle, return 'no'. "
    "4. If the image is not suitable for cattle identification or does not show a cow's muzzle, suggest 'The Cattle Registration was not found'. "
    "5. Identify all object types present in the image, and consider the context (e.g., farm, landscape) when classifying. "
    "Format the response as follows: Classification: <yes/no>, Message: <Provide an explanation>, Object Type: <object1, object2, ...>."
)



# Endpoint to analyze the uploaded image
@app.post("/analyze")
async def analyze_image(image: UploadFile = File(...)):
    try:
        # Load the uploaded image file
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))  # Open the image using PIL

        # Get the combined response from Gemini
        combined_response = get_gemini_response(combined_prompt, pil_image)

        # Initialize variables for responses
        classification_response = ""
        message_response = ""
        object_type_list = []

        # Parse the combined response safely
        if "Classification:" in combined_response and "Message:" in combined_response and "Object Type:" in combined_response:
            parts = combined_response.split("Classification:")
            if len(parts) > 1:
                classification_part = parts[1].split(", Message:")
                if len(classification_part) > 1:
                    classification_response = classification_part[0].strip()
                    message_part = classification_part[1].split(", Object Type:")
                    if len(message_part) > 1:
                        message_response = message_part[0].strip()
                        object_types = message_part[1].strip()
                        
                        # Split the object types into a list
                        object_type_list = [obj.strip() for obj in object_types.split(",")]

        # Convert "yes/no" classification to boolean
        muzzle_identified = classification_response.lower() == "yes"

        # If a cow's muzzle is identified, proceed with prediction
        if muzzle_identified:
            # Preprocess the image for prediction
            img_array = load_and_preprocess_image(np.array(pil_image))

            # Make predictions about the cattle class
            predicted_class, predicted_prob = predict_cattle_class(model, img_array)

            if predicted_class:
                registration_details = display_registration_details(predicted_class)  # Get registration details
                return {
                    "muzzle_identified": muzzle_identified,
                    "identified_objects": object_type_list,
                    "message": "The Cattle Registration was found",
                    "cattle_info": {
                        "predicted_class": predicted_class,
                        "probability": predicted_prob,
                        "details": registration_details
                    }
                }

        # Return a response if no cattle was found
        return {
            "muzzle_identified": muzzle_identified,
            "identified_objects": object_type_list,
            "message": "The Cattle Registration was not found",
            "cattle_info": {}
        }

    except Exception as e:
        # Log the error and raise an HTTP exception
        import traceback
        traceback.print_exc()  # Print the traceback for debugging
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Start the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
