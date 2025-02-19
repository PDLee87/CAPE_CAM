import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sentence_transformers import SentenceTransformer, util
import os
from tqdm import tqdm  # For progress bar

# Specify input Excel directory and file name
input_dir = "input the directory where an Excel file for CAM and CAPE texts is located"
input_file = "name of the Excel file.xlsx"
output_file = "name the output Excel file.xlsx"

# Load the Excel file
df = pd.read_excel(os.path.join(input_dir, input_file))

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Initialize Sentence Transformer model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare lists to store similarity scores
cosine_scores = []
semantic_scores = []

# Create progress bar using tqdm
for index, row in tqdm(df.iterrows(), total=len(df), desc="Calculating Similarity"):
    cam_text = str(row['CAM'])  # Ensure it's a string
    cape_text = str(row['CAPE'])  # Ensure it's a string

    # ---- Cosine Similarity using TF-IDF ----
    tfidf_matrix = tfidf_vectorizer.fit_transform([cam_text, cape_text])
    cos_sim = sklearn_cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]  # Use sklearn cosine_similarity
    cosine_scores.append(round(cos_sim, 4))

    # ---- Semantic Similarity using Transformers ----
    cam_embedding = model.encode(cam_text, convert_to_tensor=True)
    cape_embedding = model.encode(cape_text, convert_to_tensor=True)
    sem_sim = util.cos_sim(cam_embedding, cape_embedding).item()
    semantic_scores.append(round(sem_sim, 4))

# Add similarity scores to DataFrame
df['Cosine_Similarity'] = cosine_scores
df['Semantic_Similarity'] = semantic_scores

# Save the DataFrame to a new Excel file
df.to_excel(os.path.join(input_dir, output_file), index=False)

print("Similarity calculation completed and saved to:", output_file)
