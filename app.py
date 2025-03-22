import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
import gradio as gr
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the Spotify songs dataset.
    
    Args:
        file_path (str): Path to the CSV file containing Spotify songs data
        
    Returns:
        pandas.DataFrame: Preprocessed dataframe with cleaned and transformed features
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Remove rows with missing values
    df.dropna(inplace=True)
    
    # Remove unnecessary columns
    df = df.drop(['playlist_name', 'duration_ms'], axis=1)
    
    # Extract year from release date
    df['release_date'] = df['track_album_release_date'].apply(lambda x: x[:4])
    df.drop('track_album_release_date', axis=1, inplace=True)
    
    # Add remix indicator feature
    remix_keywords = ["remix", "mix", "edit", "rework", "reboot", "extended", "dub"]
    pattern = "|".join(remix_keywords) + r"|\([^)]+\s(remix|mix|edit)\)|\-\s\w+\s(remix|mix|edit)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df['remix_indicator'] = df['track_name'].str.lower().str.contains(pattern, regex=True, na=False).astype(int)
    
    # Drop album related columns as we focus on track-level features
    tracks_df = df.drop(['track_album_id', 'track_album_name'], axis=1)
    
    return tracks_df

def create_features(tracks_df):
    """
    Create feature vectors for songs by combining numerical features,
    categorical encodings, and text embeddings.
    
    Args:
        tracks_df (pandas.DataFrame): Preprocessed dataframe with track information
        
    Returns:
        tuple: (numpy.ndarray of feature vectors, updated DataFrame)
    """
    # Create artist embeddings using Word2Vec
    artists = tracks_df.groupby('playlist_id')['track_artist'].apply(lambda x: list(set(x))).tolist()
    w2vartists = Word2Vec(artists, vector_size=20, window=10, workers=4, min_count=1, seed=42)
    artist_embeddings = np.array([w2vartists.wv[artist] if artist in w2vartists.wv else np.zeros(20) 
                                 for artist in tracks_df['track_artist']])
    
    # Create track name embeddings using Word2Vec
    tracks_df['track_name_words'] = tracks_df['track_name'].str.lower().str.split()
    tracks_name_sent = tracks_df['track_name_words'].tolist()
    w2vtracks = Word2Vec(tracks_name_sent, vector_size=10, window=5, workers=4, min_count=1, seed=42)
    tracks_embedding = np.array([np.mean([w2vtracks.wv[word] for word in words if word in w2vtracks.wv] or [np.zeros(10)], axis=0) 
                                for words in tracks_name_sent])
    
    # Process numerical features
    numerical_columns = ["track_popularity", "danceability", "energy", "key", "loudness", "mode", 
                        "speechiness", "acousticness", "instrumentalness", "liveness", "valence", 
                        "tempo", "remix_indicator", "release_date"]
    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(tracks_df[numerical_columns])
    
    # One-hot encode categorical features
    genre_encode = pd.get_dummies(tracks_df['playlist_genre'])
    subgenre_encode = pd.get_dummies(tracks_df['playlist_subgenre'])
    
    # Combine all features into a single representation
    content_features = np.hstack([numeric_features, genre_encode.values, subgenre_encode.values, 
                                 tracks_embedding, artist_embeddings])
    
    return content_features, tracks_df

def recommend_tracks(seed_track_combo, tracks_df, similarity_matrix, n=5):
    """
    Recommend similar tracks based on a seed track.
    
    Args:
        seed_track_combo (str): String in format "track_name by track_artist"
        tracks_df (pandas.DataFrame): DataFrame containing track information
        similarity_matrix (numpy.ndarray): Precomputed similarity matrix
        n (int): Number of recommendations to return
        
    Returns:
        str: Formatted string of recommendations or error message
    """
    try:
        # Split the combined track and artist string
        track_name, artist = seed_track_combo.rsplit(" by ", 1)
        
        # Find the matching track in the dataset
        matching_rows = tracks_df[(tracks_df['track_name'] == track_name) & (tracks_df['track_artist'] == artist)]
        
        if len(matching_rows) == 0:
            return f"Song '{track_name}' by '{artist}' not found in dataset."
        
        # Get the index of the seed track
        seed_idx = matching_rows.index[0]
        
        # Get similarity scores for all tracks compared to the seed track
        sim_scores = similarity_matrix[seed_idx]
        
        # Find indices of the top n most similar tracks (excluding the seed track itself)
        top_indices = np.argsort(sim_scores)[::-1][1:n+1]
        
        # Get the details of the recommended tracks
        recommendations = tracks_df.iloc[top_indices][["track_name", "track_artist", "playlist_subgenre"]]
        
        # Create a formatted output with similarity scores
        result = f"Recommendations for: {track_name} by {artist}\n\n"
        
        for i, (_, row) in enumerate(recommendations.iterrows()):
            similarity = sim_scores[top_indices[i]] * 100  # Convert to percentage
            result += f"{i+1}. {row['track_name']} by {row['track_artist']}\n"
            result += f"   Genre: {row['playlist_subgenre']}\n"
            result += f"   Similarity: {similarity:.1f}%\n\n"
        
        return result
    except Exception as e:
        return f"Error finding recommendations: {str(e)}"

def gradio_recommend(track_index, num_recommendations):
    """
    Interface function for Gradio to recommend tracks based on the selected track number.
    
    Args:
        track_index (int): The index number of the selected track (1-based)
        num_recommendations (int): Number of recommendations to return
        
    Returns:
        str: Formatted string of recommendations
    """
    global tracks_df, content_sim
    
    # Check if input is valid
    try:
        track_index = int(track_index)  # Ensure the input is treated as an integer
    except ValueError:
        return f"Please enter a valid track number between 1 and {len(tracks_df)}."
    
    if track_index < 1 or track_index > len(tracks_df):
        return f"Please select a valid track number between 1 and {len(tracks_df)}."
        
    try:
        # Convert 1-based index to 0-based index
        seed_idx = track_index - 1
        
        # Get similarity scores for all tracks compared to the seed track
        sim_scores = content_sim[seed_idx]
        
        # Find indices of the top n most similar tracks (excluding the seed track itself)
        top_indices = np.argsort(sim_scores)[::-1][1:num_recommendations+1]
        
        # Get the details of the recommended tracks
        recommendations = tracks_df.iloc[top_indices][["track_name", "track_artist", "playlist_subgenre"]]
        
        # Create a formatted output with similarity scores
        seed_track = tracks_df.iloc[seed_idx]
        result = f"Recommendations for: {seed_track['track_name']} by {seed_track['track_artist']}\n\n"
        
        for i, (index, row) in enumerate(recommendations.iterrows()):
            similarity = sim_scores[top_indices[i]] * 100  # Convert to percentage
            result += f"{i+1}. {row['track_name']} by {row['track_artist']}\n"
            result += f"   Genre: {row['playlist_subgenre']}\n"
            result += f"   Similarity: {similarity:.1f}%\n\n"
        
        return result
    except Exception as e:
        return f"Error processing recommendation: {str(e)}"


# Main execution block
if __name__ == "__main__":
    # Define the data file path
    file_path = './spotify_songs.csv'
    
    # Load and preprocess the data
    tracks_df = load_and_preprocess_data(file_path)
    
    # Create feature vectors and compute similarity matrix
    content_features, tracks_df = create_features(tracks_df)
    content_sim = cosine_similarity(content_features)
    
    # Prepare the total number of tracks
    total_tracks = len(tracks_df)
    
    # Print the total number of tracks to debug
    print(f"Total number of tracks: {total_tracks}")
    
    # Create and launch the Gradio interface
    interface = gr.Interface(
        fn=gradio_recommend,
        inputs=[
            gr.Textbox(label=f"Enter a number from 1 to {total_tracks}", placeholder="Enter track number", interactive=True),
            gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Number of Recommendations")
        ],
        outputs=gr.Textbox(label="Recommended Songs", lines=10),
        title="Spotify Song Recommender",
        description="Enter a track number (from 1 to the total number of tracks) to get similar song recommendations based on audio features, genre, and artist similarity.",
        allow_flagging="never"
    )
    
    # Launch with sharing enabled for easier access
    interface.launch(share=True)
