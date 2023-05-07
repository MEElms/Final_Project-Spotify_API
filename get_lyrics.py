import pandas as pd
import lyricsgenius
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import concurrent.futures

genius = lyricsgenius.Genius("eOv_ht7f5mk0pJgWJux_uNadqykMHcbo3S8Q86QmYYxJnnso2STK1Wf5XPrm_6LT")

df = pd.read_csv("everything_playlist.csv")


# function to get lyrics for a given song and artist
def fetch_lyrics(args):
    song_title, artist_name = args
    try:
        song = genius.search_song(song_title, artist_name)
        return song.lyrics if song else None
    except:
        return None


song_artist_list = [(song, artist) for song, artist in zip(df['Song'], df['Artist'])]

# Use Thread pool to submit multiple requests at once
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(fetch_lyrics, song_artist_list)

# Add the lyrics to the dataframe
df['Lyrics'] = list(results)

# Remove any rows where the lyrics are missing
df.dropna(subset=['Lyrics'], inplace=True)

# stop_words
stop_words = set(stopwords.words('english'))
more_stop_words = ['like', 'know', "i'm", 'yeah', 'hey', 'chorus', 'get', 'fold', 'time', "ain't", 'one', 'two',
                   'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'got', 'look', 'keep', 'might',
                   "i'll", 'let', 'lyricsverse', 'prechorus', 'lyricsintro', 'postchorus', 'likechorus', 'likeverse',
                   'countdowns', 'low', 'nana', 'verse', 'see', 'huh', 'even', 'feat', "i'ma", "put", 'song', 'lyrics',
                   'want', 'wan', 'mmm', 'ooh', 'long', 'need', 'right', 'way', 'make', 'say', 'also', 'come', 'back',
                   'take', 'said', 'give', 'made', 'still', 'tell', 'old', 'thou', 'well', 'think', 'put', 'ever',
                   'last', 'may', 'people', 'new', 'left', 'things', 'came', 'always', 'around', 'another', 'really',
                   'first', 'hand', 'till', 'call', 'run', 'leave', 'show', 'find', 'thing', 'turn', 'allah', 'told',
                   'going', 'tickets']
stop_words.update(more_stop_words)


# function to count unique words
def count_words(lyrics):
    # Tokenize the lyrics
    words = word_tokenize(lyrics.lower())

    # Filter out non-wordnet words and small words
    words = [word for word in words if wordnet.synsets(word) and len(word) > 2 and word not in stop_words]

    # Count the occurrences of each unique word
    word_counts = Counter(words)

    return word_counts


# Concatenate the lyrics from all songs
all_lyrics = " ".join(df['Lyrics'].values)

# Count the unique words in the lyrics
word_counts = count_words(all_lyrics)

# Create a pandas dataframe to store the word counts
word_counts_df = pd.DataFrame(list(word_counts.items()), columns=['word', 'count'])
word_counts_df.sort_values(by='count', ascending=False, inplace=True)

# Print the 20 most common words in the lyrics
print(word_counts_df.head(40))


def to_csv(df):
    val = input("Enter File name:\n")
    df.to_csv(f"{val}.csv", index=False, encoding='utf-8')
    print(f"Done! File: {val}.csv created.")


to_csv(word_counts_df)
to_csv(df)
