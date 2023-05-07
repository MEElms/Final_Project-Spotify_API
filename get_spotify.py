import os
import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials
from time import perf_counter

start_time = perf_counter()
load_dotenv()
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

# Authentication - without user
client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# std input playlist link
#playlist_link = input("Input link to spotify playlist: \n")

# hard coded playlist link
playlist_link = "https://open.spotify.com/playlist/2FtTP5EBJy3uAyt94BpE0P?si=124522da78c749e6"

playlist_URI = playlist_link.split("/")[-1].split("?")[0]

results = sp.user_playlist_tracks(user=None, playlist_id=playlist_URI, fields=None, limit=100, offset=0, market=None)

# Initialize an empty list to store track details
tracks = []

# Iterate through all pages of playlist tracks
while results:
    # Extract track details from the current page of results
    for track in results['items']:
        # Get audio features for the track
        audio_features = sp.audio_features(track['track']['id'])[0]

        # Get genres for the artist of the track
        artist_id = track['track']['artists'][0]['id']
        artist = sp.artist(artist_id)
        genres = artist['genres']

        # Combine all track details into a dictionary
        track_details = {
            'Song': track['track']['name'],
            'Artist': track['track']['artists'][0]['name'],
            'Genres': genres,
            'Album': track['track']['album']['name'],
            'Release_Date': track['track']['album']['release_date'],
            'Duration_ms': track['track']['duration_ms'],
            'Popularity': track['track']['popularity'],
            'Danceability': audio_features['danceability'],
            'Energy': audio_features['energy'],
            'Key': audio_features['key'],
            'Loudness': audio_features['loudness'],
            'Mode': audio_features['mode'],
            'Speechiness': audio_features['speechiness'],
            'Acousticness': audio_features['acousticness'],
            'Instrumentalness': audio_features['instrumentalness'],
            'Liveness': audio_features['liveness'],
            'Valence': audio_features['valence'],
            'Tempo': audio_features['tempo'],
        }

        tracks.append(track_details)

    # Check if there are more pages of results
    if results['next']:
        results = sp.next(results)
    else:
        results = None


def to_csv(df):
    val = input("Enter File name:\n")
    df.to_csv(f"{val}.csv", index=False, encoding='utf-8')
    print(f"Done! File: {val}.csv created.")


# Create a Pandas DataFrame from the list of track details
df_playlist = pd.DataFrame(tracks)
to_csv(df_playlist)
stop_time = perf_counter()
print("Total time elapsed:", stop_time - start_time, "seconds.")
