import os
import faiss
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import torch
from tqdm import tqdm

IMAGE_SIZE = 300
VECTOR_DIM = 512
PATH = "Module_2\\Chap_9\\data"

def create_dataframe(dataset_path):
    image_paths = []
    labels = []

    for filename in os.listdir(dataset_path):
        if (filename.endswith((".jpg", ".JPG", ".png", ".jpeg"))):
            image_paths.append(os.path.join(dataset_path, filename))
            file_name = filename.split(".")[0]
            label = file_name[7:]
            labels.append(label)

    df = pd.DataFrame({"image_path": image_paths, "label": labels})
    return df

def image_to_vector(image_path):
    img = Image.open(image_path).resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img)

    # Handle grayscale image (convert to RGB)
    if (len(img_array.shape) == 2):
        img_array = np.stack([img_array] * 3, axis=-1)

    # Normalize pixel values to [0, 1]
    vector = img_array.astype("float32") / 255.0
    
    return vector.flatten()


def init_model():
    face_recognition_model = InceptionResnetV1(pretrained="vggface2").eval()

    return face_recognition_model

def init_transformer():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return transform

def extract_feature(image_path, model):
    img = Image.open(image_path).convert("RGB")
    transform = init_transformer()
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        features = model(img_tensor)
    
    return features.squeeze().numpy()

def create_index(df):
    index = faiss.IndexFlatIP(VECTOR_DIM)
    label_map = []
    face_recognition_model = init_model()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = row["image_path"]
        label = row["label"]

        try:
            features = extract_feature(image_path, face_recognition_model)
            index.add(np.array([features]))
            label_map.append(label)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    faiss.write_index(index, os.path.join(PATH, "facenet_features.index"))
    np.save(os.path.join(PATH, "facenet_label_map.npy"), np.array(label_map))

def image_to_feature(image_path, model):
    img = Image.open(image_path).convert("RGB")
    transform = init_transformer()
    img_tensor = transform(img).unsqueeze(0)    # Add batch dimension

    # Get the embedding
    with torch.no_grad():   # Disable gradient calculation
        embedding = model(img_tensor)
    
    # Return the embedding as a numpy array
    return embedding.squeeze().numpy()

def search_similar_images(query_image_path, k=5):
    # Load index and labels
    index = faiss.read_index(os.path.join(PATH, "facenet_features.index"))
    label_map = np.load(os.path.join(PATH, "facenet_label_map.npy"))

    # Convert query image to vector
    face_recognition_model = init_model()
    query_vector = image_to_feature(query_image_path, face_recognition_model)

    # Search in Faiss
    similarities, indices = index.search(np.array([query_vector]), k)

    # Get results
    results = []
    for i in range(len(indices[0])):
        employee_name = label_map[indices[0][i]]
        similarity = similarities[0][i]
        results.append((employee_name, similarity))

    return results

def display_query_and_top_matches(query_image_path, df):
    # Display query image
    query_img = Image.open(query_image_path)
    query_img = query_img.resize((IMAGE_SIZE, IMAGE_SIZE))

    plt.figure(figsize=(5, 5))
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis("off")
    plt.show()

    # Get matches
    matches = search_similar_images(query_image_path)

    # Display top matches
    plt.figure(figsize=(15, 5))
    for i, (name, similarity) in enumerate(matches):
        # Find the image path for this employee
        img_path = df[df["label"] == name]["image_path"].values[0]
        img = Image.open(img_path)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(f"{name}\nSimilarity: {similarity:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
