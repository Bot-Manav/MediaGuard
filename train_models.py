import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import glob
from PIL import Image

# Setup paths
DATASET_PATH = "mediaguard_datasets/repos"
MODELS_PATH = "models"
os.makedirs(MODELS_PATH, exist_ok=True)

def train_fake_news_model():
    print("--- Training Fake News Model ---")
    
    # Load FakeNewsNet
    fnn_path = os.path.join(DATASET_PATH, "FakeNewsNet/dataset")
    csv_files = {
        "gossipcop_fake.csv": 1,
        "gossipcop_real.csv": 0,
        "politifact_fake.csv": 1,
        "politifact_real.csv": 0
    }
    
    dfs = []
    for file, label in csv_files.items():
        path = os.path.join(fnn_path, file)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Keep only title and add label
                if 'title' in df.columns:
                    df = df[['title']].copy()
                elif 'headline' in df.columns:
                    df = df[['headline']].copy()
                    df.columns = ['title']
                else:
                    # try first text-like column
                    possible = [c for c in df.columns if 'title' in c.lower() or 'headline' in c.lower() or 'text' in c.lower()]
                    if possible:
                        df = df[[possible[0]]].copy()
                        df.columns = ['title']
                    else:
                        print(f"Skipping {path}: no title-like column")
                        continue
                df['label'] = label
                dfs.append(df)
                # Also check for augmented per-file CSVs created by augment_texts.py
                aug_path = os.path.join(DATASET_PATH, '..', 'augmented', f"aug_{file}")
                aug_path = os.path.normpath(aug_path)
                if os.path.exists(aug_path):
                    try:
                        aug_df = pd.read_csv(aug_path)
                        if 'title' in aug_df.columns:
                            aug_df = aug_df[['title']].copy()
                        elif 'headline' in aug_df.columns:
                            aug_df = aug_df[['headline']].copy()
                            aug_df.columns = ['title']
                        else:
                            possible = [c for c in aug_df.columns if 'title' in c.lower() or 'headline' in c.lower() or 'text' in c.lower()]
                            if possible:
                                aug_df = aug_df[[possible[0]]].copy()
                                aug_df.columns = ['title']
                            else:
                                aug_df = None
                        if aug_df is not None:
                            aug_df['label'] = label
                            dfs.append(aug_df)
                    except Exception as e:
                        print(f"Error loading augmented file {aug_path}: {e}")
            except Exception as e:
                print(f"Error loading {path}: {e}")
    
    # Load Liar Dataset (tsv)
    liar_path = os.path.join(DATASET_PATH, "liar_dataset")
    liar_files = ["train.tsv", "test.tsv", "valid.tsv"]
    # Liar labels binary mapping: fake (false, pants-fire, barely-true) vs real (true, mostly-true, half-true)
    liar_mapping = {
        'false': 1, 'pants-fire': 1, 'barely-true': 1,
        'true': 0, 'mostly-true': 0, 'half-true': 0
    }
    
    for file in liar_files:
        path = os.path.join(liar_path, file)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, sep='\t', header=None)
                if len(df.columns) >= 3:
                    df_subset = df[[1, 2]].copy()
                    df_subset.columns = ['label', 'title']
                    df_subset['label'] = df_subset['label'].map(liar_mapping)
                    df_subset = df_subset.dropna()
                    dfs.append(df_subset)
            except Exception as e:
                print(f"Error loading {path}: {e}")
            
    if not dfs:
        print("No fake news data found!")
        return
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.dropna()
    
    print(f"Total samples: {len(full_df)}")
    
    X = full_df['title']
    y = full_df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature extraction with N-grams and Stop Words for better context
    print("Extracting features with N-grams (1,2) and stop word removal...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 2)  # Capture word pairs like "secret cure"
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train model with standard parameters and class weighting
    print("Training balanced classifier...")
    # C=1.0 (standard regularization to avoid extreme sharpening)
    model = LogisticRegression(
        max_iter=2000, 
        C=1.0, 
        class_weight='balanced',
        solver='liblinear'
    )
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Save models
    joblib.dump(model, os.path.join(MODELS_PATH, "fake_news_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_PATH, "fake_news_vectorizer.pkl"))
    print("Fake news model saved.")

def train_deepfake_model():
    print("\n--- Training Deepfake Model (Feature Extraction + Classifier) ---")
    
    # Deepfake training: use ONLY original images (ignore augmented images)
    ff_path = os.path.join(DATASET_PATH, "FaceForensics/images")
    orig_files = []
    for root,_,fs in os.walk(ff_path):
        for f in fs:
            if f.lower().endswith(('.png','.jpg','.jpeg')):
                orig_files.append(os.path.join(root,f))

    data_orig = []
    labels_orig = []
    for img_path in orig_files:
        filename = os.path.basename(img_path).lower()
        if "original" in filename:
            label = 0
        elif any(x in filename for x in ["deepfake", "faceswap", "face2face", "neuraltextures"]):
            label = 1
        else:
            continue
        try:
            img = Image.open(img_path).convert('RGB').resize((64,64))
            img_arr = np.array(img).flatten() / 255.0
            data_orig.append(img_arr)
            labels_orig.append(label)
        except Exception:
            pass

    if len(data_orig) < 5:
        print("Not enough original image data found for training. Creating a placeholder model.")
        X = np.random.rand(20, 64*64*3)
        y = np.random.randint(0,2,20)
        print(f"Total image samples: {len(X)} (placeholder)")
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    else:
        X_orig = np.array(data_orig)
        y_orig = np.array(labels_orig)
        # split original images into train/test
        X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)
        print(f"Total original image samples: {len(X_orig)}, train size: {len(X_train)}, test size: {len(X_test)}")

    print(f"Total image samples used for training (train set size): {len(X_train)}")

    print("Training balanced deepfake classifier...")
    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight='balanced',
        solver='liblinear'
    )
    model.fit(X_train, y_train)

    # evaluate only on original test set
    try:
        y_pred = model.predict(X_test)
        print(f"Accuracy (on original test set): {accuracy_score(y_test, y_pred):.4f}")
    except Exception:
        pass

    joblib.dump(model, os.path.join(MODELS_PATH, "deepfake_model.pkl"))
    print("Deepfake model saved.")

if __name__ == "__main__":
    train_fake_news_model()
    train_deepfake_model()
