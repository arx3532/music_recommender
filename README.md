# Spotify Song Recommender

A content-based song recommendation system that suggests similar tracks based on audio features, genre, and artist similarity.

## 📖 Overview

This project implements a music recommendation system using Spotify song data. The system analyzes various attributes of songs including audio features (danceability, energy, etc.), genres, and artist relationships to find similar tracks.

Key features:
- Content-based filtering using cosine similarity
- Processing of numerical audio features, genres, and text data
- Word2Vec embeddings for artist and track name representation 
- Gradio web interface for easy interaction

## 🚀 Live Demo

Try it out: [Recommendation Model on Hugging Face Spaces](https://huggingface.co/spaces/aaazziimm/Recommendation_Model)

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spotify-recommender.git
cd spotify-recommender
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📊 Dataset

The system uses a Spotify songs dataset that should be in CSV format with the following columns:
- `track_name`: Name of the song
- `track_artist`: Artist name
- `track_popularity`: Popularity score from Spotify
- `danceability`, `energy`, `key`, `loudness`, etc.: Audio features extracted from Spotify's API
- `playlist_genre`: Main genre of the song
- `playlist_subgenre`: Subgenre classification
- `track_album_release_date`: Release date of the album

Place your dataset in the project directory as `spotify_songs.csv`.

## 🔧 Usage

Run the recommender system locally:

```bash
python app.py
```

This will launch a Gradio web interface where you can:
1. Enter a track number from the dataset
2. Choose the number of recommendations (1-10)
3. View recommended songs with similarity scores

## 🧠 How It Works

### 1. Data Preprocessing
- Removes unnecessary columns and missing values
- Extracts release year from the date
- Adds a remix indicator feature

### 2. Feature Engineering
- **Numerical Features**: Standardizes audio features like danceability, energy, etc.
- **Categorical Features**: One-hot encodes genres and subgenres
- **Text Embeddings**: 
  - Creates artist embeddings using Word2Vec
  - Generates track name embeddings from song titles

### 3. Similarity Calculation
- Combines all features into a unified vector representation
- Calculates cosine similarity between all song pairs
- Stores the similarity scores in a matrix for efficient lookup

### 4. Recommendation Generation
- Takes a seed track from the user
- Finds the most similar songs based on the precomputed similarity matrix
- Returns the top N recommendations with similarity scores

## 🛠️ Technologies Used

- **Python**: Core programming language
- **NumPy & Pandas**: Data manipulation and analysis
- **scikit-learn**: Feature preprocessing and similarity calculation
- **Gensim**: Word2Vec implementation for text embeddings
- **Gradio**: Web interface for the recommender system

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Data from Kaggle datasets
- [Gradio](https://www.gradio.app/) for the simple web interface framework
- [Gensim](https://radimrehurek.com/gensim/) for Word2Vec implementations
