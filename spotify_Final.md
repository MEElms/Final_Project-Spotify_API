---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 2
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython2
    version: 2.7.6
  nbformat: 4
  nbformat_minor: 0
---

<div class="cell code" execution_count="1"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:11:24.372172Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:11:23.597509Z&quot;}"
collapsed="true">

``` python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
import re
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import statsmodels.api as sm
```

</div>

<div class="cell code" execution_count="2"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:11:24.381670Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:11:24.374754Z&quot;}"
collapsed="false">

``` python
datafile = "/Users/mattelms/Documents/School/R-Intro-Data-Science/Spotify_Program/PySpotify/everything_playlist.csv"
df = pd.read_csv(datafile)
```

</div>

<div class="cell code" execution_count="3"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:11:24.407485Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:11:24.382844Z&quot;}"
collapsed="false">

``` python
df_clean = pd.DataFrame(df)
df_clean = df_clean.drop(columns=['Duration_ms', 'Mode'])
df_clean
```

<div class="output execute_result" execution_count="3">

                   Song         Artist  \
    0     100 Grandkids     Mac Miller   
    1             10:35         Tiësto   
    2         1, 2 Many     Luke Combs   
    3              1901        Phoenix   
    4              2055  Sleepy Hallow   
    ..              ...            ...   
    876        Daylight  David Kushner   
    877          Violet   Connor Price   
    878         Spinnin   Connor Price   
    879           Buddy   Connor Price   
    880  Jordan Belfort     Wes Walker   

                                                    Genres  \
    0                 ['hip hop', 'pittsburgh rap', 'rap']   
    1    ['big room', 'brostep', 'dutch edm', 'edm', 'h...   
    2                  ['contemporary country', 'country']   
    3    ['alternative dance', 'indie rock', 'modern ro...   
    4                        ['brooklyn drill', 'nyc rap']   
    ..                                                 ...   
    876                        ['gen z singer-songwriter']   
    877                                                 []   
    878                                                 []   
    879                                                 []   
    880                                                 []   

                                Album Release_Date  Popularity  Danceability  \
    0                        GO:OD AM   2015-09-18          65         0.735   
    1                           10:35   2022-11-03          89         0.696   
    2    What You See Is What You Get   2019-11-08          73         0.540   
    3        Wolfgang Amadeus Phoenix   2009-05-25           6         0.591   
    4                    Still Sleep?   2021-06-02          82         0.829   
    ..                            ...          ...         ...           ...   
    876                      Daylight   2023-04-14          94         0.508   
    877                Spin The Globe   2023-01-27          71         0.924   
    878                Spin The Globe   2023-01-27          74         0.765   
    879                         Buddy   2022-12-30          69         0.914   
    880                Jordan Belfort   2015-08-26          65         0.860   

         Energy  Key  Loudness  Speechiness  Acousticness  Instrumentalness  \
    0     0.749   10    -3.766       0.0874        0.3710          0.000000   
    1     0.793    8    -5.733       0.0970        0.0683          0.000004   
    2     0.821    6    -3.789       0.0873        0.0397          0.000000   
    3     0.831    0    -5.647       0.0415        0.0605          0.000047   
    4     0.512    6    -5.865       0.1870        0.4920          0.000000   
    ..      ...  ...       ...          ...           ...               ...   
    876   0.430    2    -9.475       0.0335        0.8300          0.000441   
    877   0.716    8    -7.158       0.0541        0.0177          0.000001   
    878   0.572   11    -6.153       0.1280        0.2260          0.000000   
    879   0.506    6    -5.373       0.2230        0.2040          0.000000   
    880   0.719   11    -4.325       0.3300        0.1250          0.000000   

         Liveness  Valence    Tempo  
    0      0.4700    0.373   93.718  
    1      0.1800    0.698  120.003  
    2      0.4230    0.685  148.798  
    3      0.1900    0.705  144.084  
    4      0.1200    0.638   80.511  
    ..        ...      ...      ...  
    876    0.0930    0.324  130.090  
    877    0.1280    0.480  119.973  
    878    0.1920    0.475  140.013  
    879    0.1030    0.428  140.015  
    880    0.0799    0.496  160.049  

    [881 rows x 16 columns]

</div>

</div>

<div class="cell markdown" collapsed="false">

# MAJOR BATHING OF DATA AHEAD

</div>

<div class="cell markdown" collapsed="false">

### Cleaned tempo and loudness data to be on the same scale as the rest of the attributes (0-1.0)

</div>

<div class="cell code" execution_count="4"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:11:24.407665Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:11:24.402384Z&quot;}"
collapsed="false">

``` python
# Min - Max for tempo
min_tempo = df_clean['Tempo'].min()
max_tempo = df_clean['Tempo'].max()

# Scale the values in tempo to range from 0 to 1
df_clean['Tempo'] = (df_clean['Tempo'] - min_tempo) / (max_tempo - min_tempo)

# Min - Max for loudness
min_loudness = df_clean['Loudness'].min()
max_loudness = df_clean['Loudness'].max()

# Scale the values in loudness to range from 0 to 1
df_clean['Loudness'] = (df_clean['Loudness'] - min_loudness) / (max_loudness - min_loudness)
```

</div>

<div class="cell markdown" collapsed="false">

### Change Release Date to an actual date

</div>

<div class="cell code" execution_count="5"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:11:24.417019Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:11:24.414391Z&quot;}"
collapsed="false">

``` python
# Convert 'Release Date' to an actual date
df_clean['Release_Date'] = pd.to_datetime(df_clean['Release_Date'])
df_clean
```

<div class="output execute_result" execution_count="5">

                   Song         Artist  \
    0     100 Grandkids     Mac Miller   
    1             10:35         Tiësto   
    2         1, 2 Many     Luke Combs   
    3              1901        Phoenix   
    4              2055  Sleepy Hallow   
    ..              ...            ...   
    876        Daylight  David Kushner   
    877          Violet   Connor Price   
    878         Spinnin   Connor Price   
    879           Buddy   Connor Price   
    880  Jordan Belfort     Wes Walker   

                                                    Genres  \
    0                 ['hip hop', 'pittsburgh rap', 'rap']   
    1    ['big room', 'brostep', 'dutch edm', 'edm', 'h...   
    2                  ['contemporary country', 'country']   
    3    ['alternative dance', 'indie rock', 'modern ro...   
    4                        ['brooklyn drill', 'nyc rap']   
    ..                                                 ...   
    876                        ['gen z singer-songwriter']   
    877                                                 []   
    878                                                 []   
    879                                                 []   
    880                                                 []   

                                Album Release_Date  Popularity  Danceability  \
    0                        GO:OD AM   2015-09-18          65         0.735   
    1                           10:35   2022-11-03          89         0.696   
    2    What You See Is What You Get   2019-11-08          73         0.540   
    3        Wolfgang Amadeus Phoenix   2009-05-25           6         0.591   
    4                    Still Sleep?   2021-06-02          82         0.829   
    ..                            ...          ...         ...           ...   
    876                      Daylight   2023-04-14          94         0.508   
    877                Spin The Globe   2023-01-27          71         0.924   
    878                Spin The Globe   2023-01-27          74         0.765   
    879                         Buddy   2022-12-30          69         0.914   
    880                Jordan Belfort   2015-08-26          65         0.860   

         Energy  Key  Loudness  Speechiness  Acousticness  Instrumentalness  \
    0     0.749   10  0.811862       0.0874        0.3710          0.000000   
    1     0.793    8  0.700227       0.0970        0.0683          0.000004   
    2     0.821    6  0.810556       0.0873        0.0397          0.000000   
    3     0.831    0  0.705108       0.0415        0.0605          0.000047   
    4     0.512    6  0.692736       0.1870        0.4920          0.000000   
    ..      ...  ...       ...          ...           ...               ...   
    876   0.430    2  0.487855       0.0335        0.8300          0.000441   
    877   0.716    8  0.619353       0.0541        0.0177          0.000001   
    878   0.572   11  0.676390       0.1280        0.2260          0.000000   
    879   0.506    6  0.720658       0.2230        0.2040          0.000000   
    880   0.719   11  0.780136       0.3300        0.1250          0.000000   

         Liveness  Valence     Tempo  
    0      0.4700    0.373  0.280841  
    1      0.1800    0.698  0.444883  
    2      0.4230    0.685  0.624590  
    3      0.1900    0.705  0.595171  
    4      0.1200    0.638  0.198417  
    ..        ...      ...       ...  
    876    0.0930    0.324  0.507835  
    877    0.1280    0.480  0.444696  
    878    0.1920    0.475  0.569764  
    879    0.1030    0.428  0.569777  
    880    0.0799    0.496  0.694807  

    [881 rows x 16 columns]

</div>

</div>

<div class="cell markdown" collapsed="false">

### Remove Duplicated songs in the playlist if there are any

</div>

<div class="cell code" execution_count="6"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:11:24.422116Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:11:24.418375Z&quot;}"
collapsed="false">

``` python
df_clean = df_clean.drop_duplicates(subset=['Song', 'Artist'], keep='first')
df_clean = df_clean.sort_values('Song').reset_index(drop=True)
```

</div>

<div class="cell markdown" collapsed="false">

###### 6 songs were dropped as duplicates!

</div>

<div class="cell code" execution_count="7"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:11:24.426550Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:11:24.424111Z&quot;}"
collapsed="false">

``` python
df_clean.shape
```

<div class="output execute_result" execution_count="7">

    (875, 16)

</div>

</div>

<div class="cell markdown" collapsed="false">

### Cleaned the Genres column to have only the first genre of the "list"

</div>

<div class="cell code" execution_count="8"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:11:24.553535Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:11:24.523344Z&quot;}"
collapsed="false">

``` python
df_clean['Genres'] = df_clean['Genres'].replace(to_replace=r"[\[\]']", value='', regex=True)
df_clean['Genre'] = df_clean['Genres'].str.split(',').str[0]
df_clean = df_clean.drop('Genres', axis=1)
df_clean
df_clean = df_clean[
    ['Song', 'Artist', 'Genre', 'Album', 'Release_Date', 'Popularity', 'Danceability', 'Energy', 'Key', 'Loudness',
     'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']]
df_clean
```

<div class="output execute_result" execution_count="8">

                                         Song               Artist  \
    0                          'Til You Can't         Cody Johnson   
    1                               1, 2 Many           Luke Combs   
    2                           100 Grandkids           Mac Miller   
    3                                   10:35               Tiësto   
    4                                    1901              Phoenix   
    ..                                    ...                  ...   
    870  my ex's best friend (with blackbear)    Machine Gun Kelly   
    871                            ocean eyes        Billie Eilish   
    872                                 oops!           Yung Gravy   
    873                                   oui              Jeremih   
    874   overwhelmed - Chri$tian Gate$ remix  Royal & the Serpent   

                         Genre                         Album Release_Date  \
    0    classic texas country        Human The Double Album   2021-10-08   
    1     contemporary country  What You See Is What You Get   2019-11-08   
    2                  hip hop                      GO:OD AM   2015-09-18   
    3                 big room                         10:35   2022-11-03   
    4        alternative dance      Wolfgang Amadeus Phoenix   2009-05-25   
    ..                     ...                           ...          ...   
    870           ohio hip hop        Tickets To My Downfall   2020-09-25   
    871                art pop             Summer Heartbreak   2021-08-13   
    872               meme rap                      Gasanova   2020-10-02   
    873            chicago rap        Late Nights: The Album   2015-12-04   
    874                  alt z     overwhelmed (the remixes)   2021-02-05   

         Popularity  Danceability  Energy  Key  Loudness  Speechiness  \
    0            75         0.501   0.815    1  0.749489       0.0436   
    1            73         0.540   0.821    6  0.810556       0.0873   
    2            65         0.735   0.749   10  0.811862       0.0874   
    3            89         0.696   0.793    8  0.700227       0.0970   
    4             6         0.591   0.831    0  0.705108       0.0415   
    ..          ...           ...     ...  ...       ...          ...   
    870          77         0.731   0.675    5  0.734222       0.0434   
    871           0         0.358   0.372    4  0.590636       0.0464   
    872          69         0.886   0.743    6  0.639841       0.0812   
    873          77         0.418   0.724    5  0.814245       0.0964   
    874          67         0.619   0.502    0  0.566061       0.2490   

         Acousticness  Instrumentalness  Liveness  Valence     Tempo  
    0         0.05130          0.000000    0.1060    0.460  0.695044  
    1         0.03970          0.000000    0.4230    0.685  0.624590  
    2         0.37100          0.000000    0.4700    0.373  0.280841  
    3         0.06830          0.000004    0.1800    0.698  0.444883  
    4         0.06050          0.000047    0.1900    0.705  0.595171  
    ..            ...               ...       ...      ...       ...  
    870       0.00473          0.000000    0.1410    0.298  0.475689  
    871       0.81000          0.053900    0.0877    0.160  0.599845  
    872       0.03580          0.000198    0.0511    0.941  0.519144  
    873       0.21300          0.000000    0.1120    0.604  0.185998  
    874       0.17800          0.000000    0.1550    0.380  0.783097  

    [875 rows x 16 columns]

</div>

</div>

<div class="cell markdown" collapsed="false">

### The Genres were very... vast; meaning the genres used very specific sub-genres rather than common genre categories. This code fixes this problem.

##### ChatGPT was used to turn this process into a 5 minute task. By printing all the unique genres, I was able to tell ChatGPT to turn that list into a consolidated genres; Country, Rock, Rap, Pop, Metal, etc.

</div>

<div class="cell code" execution_count="9"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:11:25.085922Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:11:25.060540Z&quot;}"
collapsed="false">

``` python
# ChatGPT gets an award for this... JESUS
genre_map = {
    'classic texas country': 'Country',
    'contemporary country': 'Country',
    'country pop': 'Country',
    'classic oklahoma country': 'Country',
    'australian country': 'Country',
    'country': 'Country',
    'country rock': 'Country',
    'classic country pop': 'Country',
    'arkansas country': 'Country',
    'modern southern rock': 'Country',
    'country rap': 'Country',
    'redneck': 'Country',
    'modern country pop': 'Country',

    'big room': 'EDM',
    'edm': 'EDM',
    'brostep': 'EDM',
    'complextro': 'EDM',
    'dutch house': 'EDM',
    'europop': 'EDM',
    'dancefloor dnb': 'EDM',
    'deep pop edm': 'EDM',
    'dutch edm': 'EDM',
    'australian dance': 'EDM',
    'classic hardstyle': 'EDM',
    'aussietronica': 'EDM',
    'filter house': 'EDM',
    'danish electronic': 'EDM',
    'canadian electronic': 'EDM',
    'viral trap': 'EDM',
    'downtempo': 'EDM',
    'uk dance': 'EDM',
    'tropical house': 'EDM',
    'future funk': 'EDM',

    'alternative dance': 'Alternative',
    'modern indie pop': 'Alternative',
    'indie pop rap': 'Alternative',
    'pov: indie': 'Alternative',
    'indie rock italiano': 'Alternative',
    'indietronica': 'Alternative',
    'alternative rock': 'Alternative',
    'modern alternative rock': 'Alternative',
    'british alternative rock': 'Alternative',
    'alternative hip hop': 'Alternative',
    'neon pop punk': 'Alternative',
    'modern alternative pop': 'Alternative',
    'hopebeat': 'Alternative',
    'alt z': 'Alternative',
    'shimmer psych': 'Alternative',
    'indie poptimism': 'Alternative',
    'indie pop': 'Alternative',
    'pixie': 'Alternative',
    'french shoegaze': 'Alternative',
    'icelandic indie': 'Alternative',
    'canadian indie': 'Alternative',
    'escape room': 'Alternative',
    'indie rock': 'Alternative',
    'emo': 'Alternative',
    'brooklyn indie': 'Alternative',
    'bath indie': 'Alternative',

    'pop': 'Pop',
    'dance pop': 'Pop',
    'danish pop': 'Pop',
    'adult standards': 'Pop',
    'lilith': 'Pop',
    'folk-pop': 'Pop',
    'candy pop': 'Pop',
    'social media pop': 'Pop',
    'pop rock': 'Pop',
    'new romantic': 'Pop',
    'la pop': 'Pop',
    'australian pop': 'Pop',
    'pop dance': 'Pop',
    'pop punk': 'Pop',
    'post-teen pop': 'Pop',
    'viral pop': 'Pop',
    'pop rap': 'Pop',
    'art pop': 'Pop',
    'gauze pop': 'Pop',
    'canadian pop': 'Pop',
    'bubblegum pop': 'Pop',
    'girl group': 'Pop',
    'electropop': 'Pop',
    'neo mellow': 'Pop',
    'bossbeat': 'Pop',
    'ccm': 'Pop',
    'boy band': 'Pop',
    'swedish pop': 'Pop',
    'baroque pop': 'Pop',
    'karaoke': 'Pop',
    'german pop': 'Pop',
    'chamber pop': 'Pop',
    'scandipop': 'Pop',
    'canadian contemporary r&b': 'Pop',
    'acoustic pop': 'Pop',
    'disco': 'Pop',
    'bedroom pop': 'Pop',

    'alternative metal': 'Metal',
    'nu metal': 'Metal',
    'metalcore': 'Metal',
    'melodic metalcore': 'Metal',
    'american metalcore': 'Metal',
    'prog metal': 'Metal',
    'comic metal': 'Metal',

    'rock': 'Rock',
    'piano rock': 'Rock',
    'modern rock': 'Rock',
    'classic rock': 'Rock',
    'dance rock': 'Rock',
    'album rock': 'Rock',
    'garage rock': 'Rock',
    'blues rock': 'Rock',
    'modern folk rock': 'Rock',
    'celtic rock': 'Rock',
    'classic canadian rock': 'Rock',
    'permanent wave': 'Rock',
    'kentucky indie': 'Rock',
    'modern blues rock': 'Rock',
    'deathgrass': 'Rock',
    'canadian punk': 'Rock',
    'beatlesque': 'Rock',

    'world': 'World',
    'flamenco': 'World',
    'shanty': 'World',
    'mariachi': 'World',
    'reggae cover': 'World',
    'reggaeton': 'World',
    'movie tunes': 'World',
    'orchestral soundtrack': 'World',
    'latin pop': 'World',
    'black americana': 'World',

    'hip hop': 'Hip hop',
    'australian hip hop': 'Hip hop',
    'atl hip hop': 'Hip hop',
    'ohio hip hop': 'Hip hop',
    'pittsburgh indie': 'Hip hop',
    'miami hip hop': 'Hip hop',
    'east coast hip hop': 'Hip hop',
    'la indie': 'Hip hop',
    'lgbtq+ hip hop': 'Hip hop',
    'alberta country': 'Hip hop',
    'canadian hip hop': 'Hip hop',
    'uk contemporary r&b': 'Hip hop',
    'memphis soul': 'Hip hop',

    'rap': 'Rap',
    'trap': 'Rap',
    'cali rap': 'Rap',
    'chicago rap': 'Rap',
    'new jersey rap': 'Rap',
    'dfw rap': 'Rap',
    'brooklyn drill': 'Rap',
    'conscious hip hop': 'Rap',
    'detroit hip hop': 'Rap',
    'chicago house': 'Rap',
    'sad rap': 'Rap',
    'metropopolis': 'Rap',
    'gen z singer-songwriter': 'Rap',
    'rap rock': 'Rap',
    'deep underground hip hop': 'Rap',
    'viral rap': 'Rap',
    'meme rap': 'Rap',
    'dirty south rap': 'Rap',
    'comedy rap': 'Rap',
    'maga rap': 'Rap',
    'double drumming': 'Rap',
    'banjo': 'Rap',
    'melodic rap': 'Rap',

    'soul': 'Soul',
    'neo soul': 'Soul',
    'classic soul': 'Soul',
    'british soul': 'Soul',
    'r&b': 'Soul',
    'bedroom soul': 'Soul'
}
df_clean['Genre'] = df_clean['Genre'].replace(genre_map)
print(df_clean['Genre'].unique())
df_clean
```

<div class="output stream stdout">

    ['Country' 'Hip hop' 'EDM' 'Alternative' 'Rap' 'Pop' 'Metal' 'Rock' ''
     'World' 'Soul']

</div>

<div class="output execute_result" execution_count="9">

                                         Song               Artist        Genre  \
    0                          'Til You Can't         Cody Johnson      Country   
    1                               1, 2 Many           Luke Combs      Country   
    2                           100 Grandkids           Mac Miller      Hip hop   
    3                                   10:35               Tiësto          EDM   
    4                                    1901              Phoenix  Alternative   
    ..                                    ...                  ...          ...   
    870  my ex's best friend (with blackbear)    Machine Gun Kelly      Hip hop   
    871                            ocean eyes        Billie Eilish          Pop   
    872                                 oops!           Yung Gravy          Rap   
    873                                   oui              Jeremih          Rap   
    874   overwhelmed - Chri$tian Gate$ remix  Royal & the Serpent  Alternative   

                                Album Release_Date  Popularity  Danceability  \
    0          Human The Double Album   2021-10-08          75         0.501   
    1    What You See Is What You Get   2019-11-08          73         0.540   
    2                        GO:OD AM   2015-09-18          65         0.735   
    3                           10:35   2022-11-03          89         0.696   
    4        Wolfgang Amadeus Phoenix   2009-05-25           6         0.591   
    ..                            ...          ...         ...           ...   
    870        Tickets To My Downfall   2020-09-25          77         0.731   
    871             Summer Heartbreak   2021-08-13           0         0.358   
    872                      Gasanova   2020-10-02          69         0.886   
    873        Late Nights: The Album   2015-12-04          77         0.418   
    874     overwhelmed (the remixes)   2021-02-05          67         0.619   

         Energy  Key  Loudness  Speechiness  Acousticness  Instrumentalness  \
    0     0.815    1  0.749489       0.0436       0.05130          0.000000   
    1     0.821    6  0.810556       0.0873       0.03970          0.000000   
    2     0.749   10  0.811862       0.0874       0.37100          0.000000   
    3     0.793    8  0.700227       0.0970       0.06830          0.000004   
    4     0.831    0  0.705108       0.0415       0.06050          0.000047   
    ..      ...  ...       ...          ...           ...               ...   
    870   0.675    5  0.734222       0.0434       0.00473          0.000000   
    871   0.372    4  0.590636       0.0464       0.81000          0.053900   
    872   0.743    6  0.639841       0.0812       0.03580          0.000198   
    873   0.724    5  0.814245       0.0964       0.21300          0.000000   
    874   0.502    0  0.566061       0.2490       0.17800          0.000000   

         Liveness  Valence     Tempo  
    0      0.1060    0.460  0.695044  
    1      0.4230    0.685  0.624590  
    2      0.4700    0.373  0.280841  
    3      0.1800    0.698  0.444883  
    4      0.1900    0.705  0.595171  
    ..        ...      ...       ...  
    870    0.1410    0.298  0.475689  
    871    0.0877    0.160  0.599845  
    872    0.0511    0.941  0.519144  
    873    0.1120    0.604  0.185998  
    874    0.1550    0.380  0.783097  

    [875 rows x 16 columns]

</div>

</div>

<div class="cell markdown" collapsed="false">

#### The Spotify API is not perfect and not every song has a genre, therefore, I needed to manually clean the data.

</div>

<div class="cell code" execution_count="10"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:11:25.982595Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:11:25.959682Z&quot;}"
collapsed="false">

``` python
# Manual Cleaning :( No AI help on this one guys
df_clean.loc[df_clean['Artist'] == 'Connor Price', 'Genre'] = 'Rap'
df_clean.loc[df_clean['Artist'] == 'Kyndal Inskeep', 'Genre'] = 'Country'
df_clean.loc[df_clean['Artist'] == '1 Hour Band', 'Genre'] = 'Pop'
df_clean.loc[df_clean['Artist'] == 'Alice Merton', 'Genre'] = 'Pop'
df_clean.loc[df_clean['Artist'] == 'Mashd N Kutcher', 'Genre'] = 'EDM'
df_clean.loc[df_clean['Artist'] == 'L.B. One', 'Genre'] = 'Rock'
df_clean.loc[df_clean['Artist'] == 'Wes Walker', 'Genre'] = 'Rap'
df_clean.loc[df_clean['Artist'] == 'Nic D', 'Genre'] = 'Pop'
df_clean.loc[df_clean['Artist'] == 'Nicky Youre', 'Genre'] = 'Pop'
df_clean.loc[df_clean['Artist'] == 'Everlast', 'Genre'] = 'Hip hop'
df_clean.loc[df_clean['Artist'] == 'Nico & Vinz', 'Genre'] = 'Pop'
df_clean.loc[df_clean['Artist'] == 'NEFFEX', 'Genre'] = 'Metal'
df_clean.loc[df_clean['Artist'] == 'Superstar Pride', 'Genre'] = 'Rap'
df_clean.loc[df_clean['Artist'] == 'ROSE BEAT', 'Genre'] = 'EDM'
df_clean.loc[df_clean['Artist'] == 'Drake White', 'Genre'] = 'Country'
df_clean.loc[df_clean['Artist'] == 'Justice Moses', 'Genre'] = 'Hip hop'
df_clean.loc[df_clean['Artist'] == 'Lakeview', 'Genre'] = 'Country'
df_clean.loc[df_clean['Artist'] == 'HIIT BPM', 'Genre'] = 'Country'
df_clean.loc[df_clean['Artist'] == 'PLVTINUM', 'Genre'] = 'Rap'
df_clean.loc[df_clean['Artist'] == 'Redfoo', 'Genre'] = 'EDM'
df_clean.loc[df_clean['Artist'] == 'Brooks Jefferson', 'Genre'] = 'Country'
df_clean.loc[df_clean['Artist'] == 'Two Friends', 'Genre'] = 'EDM'
df_clean.loc[df_clean['Artist'] == 'Niko Moon', 'Genre'] = 'Country'
df_clean.loc[df_clean['Artist'] == 'Caleb Mills', 'Genre'] = 'Country'
df_clean.loc[df_clean['Artist'] == 'Mark Ronson', 'Genre'] = 'Pop'
df_clean.loc[df_clean['Artist'] == 'John Harvie', 'Genre'] = 'Rock'
df_clean.loc[df_clean['Artist'] == 'Tangerine Kitty', 'Genre'] = 'Pop'
df_clean.loc[df_clean['Artist'] == 'Social House', 'Genre'] = 'Hip hop'
df_clean.loc[df_clean['Artist'] == 'TWISTED', 'Genre'] = 'EDM'
df_clean.loc[df_clean['Artist'] == 'ScurtDae', 'Genre'] = 'Hip hop'
df_clean.loc[df_clean['Song'] == 'Led', 'Genre'] = 'Rap'
df_clean
```

<div class="output execute_result" execution_count="10">

                                         Song               Artist        Genre  \
    0                          'Til You Can't         Cody Johnson      Country   
    1                               1, 2 Many           Luke Combs      Country   
    2                           100 Grandkids           Mac Miller      Hip hop   
    3                                   10:35               Tiësto          EDM   
    4                                    1901              Phoenix  Alternative   
    ..                                    ...                  ...          ...   
    870  my ex's best friend (with blackbear)    Machine Gun Kelly      Hip hop   
    871                            ocean eyes        Billie Eilish          Pop   
    872                                 oops!           Yung Gravy          Rap   
    873                                   oui              Jeremih          Rap   
    874   overwhelmed - Chri$tian Gate$ remix  Royal & the Serpent  Alternative   

                                Album Release_Date  Popularity  Danceability  \
    0          Human The Double Album   2021-10-08          75         0.501   
    1    What You See Is What You Get   2019-11-08          73         0.540   
    2                        GO:OD AM   2015-09-18          65         0.735   
    3                           10:35   2022-11-03          89         0.696   
    4        Wolfgang Amadeus Phoenix   2009-05-25           6         0.591   
    ..                            ...          ...         ...           ...   
    870        Tickets To My Downfall   2020-09-25          77         0.731   
    871             Summer Heartbreak   2021-08-13           0         0.358   
    872                      Gasanova   2020-10-02          69         0.886   
    873        Late Nights: The Album   2015-12-04          77         0.418   
    874     overwhelmed (the remixes)   2021-02-05          67         0.619   

         Energy  Key  Loudness  Speechiness  Acousticness  Instrumentalness  \
    0     0.815    1  0.749489       0.0436       0.05130          0.000000   
    1     0.821    6  0.810556       0.0873       0.03970          0.000000   
    2     0.749   10  0.811862       0.0874       0.37100          0.000000   
    3     0.793    8  0.700227       0.0970       0.06830          0.000004   
    4     0.831    0  0.705108       0.0415       0.06050          0.000047   
    ..      ...  ...       ...          ...           ...               ...   
    870   0.675    5  0.734222       0.0434       0.00473          0.000000   
    871   0.372    4  0.590636       0.0464       0.81000          0.053900   
    872   0.743    6  0.639841       0.0812       0.03580          0.000198   
    873   0.724    5  0.814245       0.0964       0.21300          0.000000   
    874   0.502    0  0.566061       0.2490       0.17800          0.000000   

         Liveness  Valence     Tempo  
    0      0.1060    0.460  0.695044  
    1      0.4230    0.685  0.624590  
    2      0.4700    0.373  0.280841  
    3      0.1800    0.698  0.444883  
    4      0.1900    0.705  0.595171  
    ..        ...      ...       ...  
    870    0.1410    0.298  0.475689  
    871    0.0877    0.160  0.599845  
    872    0.0511    0.941  0.519144  
    873    0.1120    0.604  0.185998  
    874    0.1550    0.380  0.783097  

    [875 rows x 16 columns]

</div>

</div>

<div class="cell markdown" collapsed="false">

## This is the cleaned Dataset ready to be analyzed

</div>

<div class="cell code" execution_count="11"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:11:27.055600Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:11:27.037332Z&quot;}"
collapsed="false">

``` python
df_clean
```

<div class="output execute_result" execution_count="11">

                                         Song               Artist        Genre  \
    0                          'Til You Can't         Cody Johnson      Country   
    1                               1, 2 Many           Luke Combs      Country   
    2                           100 Grandkids           Mac Miller      Hip hop   
    3                                   10:35               Tiësto          EDM   
    4                                    1901              Phoenix  Alternative   
    ..                                    ...                  ...          ...   
    870  my ex's best friend (with blackbear)    Machine Gun Kelly      Hip hop   
    871                            ocean eyes        Billie Eilish          Pop   
    872                                 oops!           Yung Gravy          Rap   
    873                                   oui              Jeremih          Rap   
    874   overwhelmed - Chri$tian Gate$ remix  Royal & the Serpent  Alternative   

                                Album Release_Date  Popularity  Danceability  \
    0          Human The Double Album   2021-10-08          75         0.501   
    1    What You See Is What You Get   2019-11-08          73         0.540   
    2                        GO:OD AM   2015-09-18          65         0.735   
    3                           10:35   2022-11-03          89         0.696   
    4        Wolfgang Amadeus Phoenix   2009-05-25           6         0.591   
    ..                            ...          ...         ...           ...   
    870        Tickets To My Downfall   2020-09-25          77         0.731   
    871             Summer Heartbreak   2021-08-13           0         0.358   
    872                      Gasanova   2020-10-02          69         0.886   
    873        Late Nights: The Album   2015-12-04          77         0.418   
    874     overwhelmed (the remixes)   2021-02-05          67         0.619   

         Energy  Key  Loudness  Speechiness  Acousticness  Instrumentalness  \
    0     0.815    1  0.749489       0.0436       0.05130          0.000000   
    1     0.821    6  0.810556       0.0873       0.03970          0.000000   
    2     0.749   10  0.811862       0.0874       0.37100          0.000000   
    3     0.793    8  0.700227       0.0970       0.06830          0.000004   
    4     0.831    0  0.705108       0.0415       0.06050          0.000047   
    ..      ...  ...       ...          ...           ...               ...   
    870   0.675    5  0.734222       0.0434       0.00473          0.000000   
    871   0.372    4  0.590636       0.0464       0.81000          0.053900   
    872   0.743    6  0.639841       0.0812       0.03580          0.000198   
    873   0.724    5  0.814245       0.0964       0.21300          0.000000   
    874   0.502    0  0.566061       0.2490       0.17800          0.000000   

         Liveness  Valence     Tempo  
    0      0.1060    0.460  0.695044  
    1      0.4230    0.685  0.624590  
    2      0.4700    0.373  0.280841  
    3      0.1800    0.698  0.444883  
    4      0.1900    0.705  0.595171  
    ..        ...      ...       ...  
    870    0.1410    0.298  0.475689  
    871    0.0877    0.160  0.599845  
    872    0.0511    0.941  0.519144  
    873    0.1120    0.604  0.185998  
    874    0.1550    0.380  0.783097  

    [875 rows x 16 columns]

</div>

</div>

<div class="cell markdown" collapsed="false">

### Question: Percentage of Genres in the Playlist

##### Genre Breakdown for entire playlist

</div>

<div class="cell code" execution_count="12"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:11:28.075091Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:11:27.938074Z&quot;}"
collapsed="false">

``` python
genre_counts = df_clean['Genre'].value_counts()

fig, ax = plt.subplots(figsize=(10, 8))


def autopct_format(pct):
    return f'{pct:.1f}%' if pct > 0 else ''


wedges, texts, autotexts = ax.pie(
    genre_counts,
    labels=genre_counts.index,
    autopct=autopct_format,
    startangle=180,
    pctdistance=0.75
)

ax.axis('equal')
plt.setp(texts, fontsize=12)
plt.setp(autotexts, fontsize=10)
centre_circle = plt.Circle((0, 0), 0.25, fc='white')
fig.gca().add_artist(centre_circle)
plt.legend(title="Genres", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.title("Percentage of Songs in Each Genre")

# Display the chart
plt.show()
```

<div class="output display_data">

![](bcf7a023b131461a7b08be41f0866007f9f1e139.png)

</div>

</div>

<div class="cell markdown" collapsed="false">

### Question: Top 25 Years with the Highest Number of Songs Released in Playlist

#### Song Release year breakdown for entire playlist

</div>

<div class="cell code" execution_count="13"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:11:28.990685Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:11:28.838417Z&quot;}"
collapsed="false">

``` python
# Extract the year from the 'Date' column and create a new 'Year' column
year = df_clean['Release_Date'].dt.year

# Group the dataframe by 'Year' and count the instances
year_counts = year.value_counts().reset_index()
year_counts.columns = ['Year', 'Count']
year_counts = year_counts.sort_values(by='Count', ascending=False)
# Select the top 25 years
top_25_years = year_counts.head(25)

# Create a bar graph using Seaborn
plt.figure(figsize=(15, 6))
sns.barplot(x='Year', y='Count', data=top_25_years, palette='pastel')
plt.title('Top 25 Years with the Highest Number of Songs Released in Playlist')
plt.xlabel('Year')
plt.ylabel('Count')

plt.show()
```

<div class="output display_data">

![](1157a7e24ae2230bfe1611f52a8783c9224d4360.png)

</div>

</div>

<div class="cell markdown" collapsed="false">

This graph shows my top 25 years with the highest number of songs
released in the playlist. My most common years were between 2014-2021!

</div>

<div class="cell markdown" collapsed="false">

### Question: Top 10 Artists with the Highest Number of Songs in the Playlist

#### Artist breakdown for entire playlist

</div>

<div class="cell code" execution_count="19"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:15:48.928826Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:15:48.787095Z&quot;}"
collapsed="false">

``` python
top_artists = df_clean['Artist'].value_counts().head(10)

top_artists_df = top_artists.reset_index()
top_artists_df.columns = ['artist', 'frequency']

# Create a color map
colors = sns.color_palette("coolwarm_r", n_colors=len(top_artists))

# Create the bar chart
plt.figure(figsize=(15, 6))
sns.barplot(x='artist', y='frequency', data=top_artists_df, palette=colors)
plt.title('Top 10 Artists in My Spotify Playlist')
plt.xlabel('Artist')
plt.ylabel('Frequency')
plt.xticks(rotation=60)
plt.show()
```

<div class="output display_data">

![](5428a8ae1e7c752b77f1fd8b85760ce020184c40.png)

</div>

</div>

<div class="cell markdown" collapsed="false">

This graph shows my top 10 artists in my Spotify playlist. It also shows
how diverse my musical taste is and it shows that I don't like an artist
just because they are an artist. Morgan Wallen has over 100 songs, but I
only liked roughly a quarter of his songs.

</div>

<div class="cell markdown" collapsed="false">

### Question: Determine how genres effect different attributes in the Spotify Playlist

#### Graph to show the average of song attributes by genre

</div>

<div class="cell code" execution_count="20"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:15:51.298296Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:15:51.156482Z&quot;}"
collapsed="false">

``` python
grouped_by_genre = df_clean.groupby('Genre')
attributes = ['Danceability', 'Energy', 'Valence', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Tempo', 'Key']
mean_by_genre = grouped_by_genre[attributes].mean()

# Create a new DataFrame with the mean values of the song attributes and genre labels
selected_data = mean_by_genre.reset_index()

selected_data['Key'] = selected_data['Key'] / 10

# Create the parallel coordinates plot
plt.figure(figsize=(15, 6))
ax = parallel_coordinates(selected_data, 'Genre', colormap='viridis')

# Set the legend location
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.title('Parallel Coordinates of Song Attributes by Genre')
plt.xticks(rotation=60)
plt.show()
```

<div class="output display_data">

![](dff97321e99c486ae87eb4ce046cf8b36b91dbf9.png)

</div>

</div>

<div class="cell markdown" collapsed="false">

This graph groups all the songs by genres and analyzes the averages for
each attribute. This not only shows the different genres and how they
might be different because of the attributes but also shows general
trend of my music as a whole!

</div>

<div class="cell markdown" collapsed="false">

## Using another python script, get_lyrics.py, I was able to create two CSVs one with lyrics and one with word counts.

</div>

<div class="cell code" execution_count="24"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:20:10.170185Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:20:10.153482Z&quot;}"
collapsed="false">

``` python
datafile = "/Users/mattelms/Documents/School/R-Intro-Data-Science/Spotify_Program/PySpotify/word_counts.csv"
df_lyricsCount = pd.read_csv(datafile)
df_lyricsCountTest = df_lyricsCount
```

</div>

<div class="cell markdown" collapsed="false">

#### Question: Using NLP to get the top 10 Words used in Lyrics in My Spotify Playlist

</div>

<div class="cell code" execution_count="50"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:29:18.632579Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:29:18.525614Z&quot;}"
collapsed="false">

``` python
# Create a color map with green and blue colors
colors = sns.color_palette("coolwarm_r", n_colors=10)

# Create the bar chart
plt.figure(figsize=(15, 6))
sns.barplot(y='word', x='count', data=df_lyricsCount.head(10), palette=colors)
plt.title('Top 10 Words used in Lyrics in My Spotify Playlist')
plt.ylabel('Word')
plt.xlabel('Frequency')
plt.show()
```

<div class="output display_data">

![](9d65ef93b61c15bf0f0abc3be05b392edc060503.png)

</div>

</div>

<div class="cell markdown" collapsed="false">

This graph depicts the top ten words used in lyrics in my Spotify
playlist. I find it an interesting way to use NLP to discover the type
of music I listen to.

</div>

<div class="cell markdown" collapsed="false">

# Disclaimer:

#### This is to explore Natural Language Processing, I feel as 'Bad' words are part of natural language, they should be processed as well. Artists add 'bad' words to music to convey how they truly feel. My time in the service was littered with the use of foul language, however, it was natural and ingrained in us. I felt it would be an interesting perspective to look at all my favorite musics lyrics, but I would be doing an injustice to not analyze what might be the most important, thought-provoking words artists use in their music and why I listen to them. I will censor any foul words as this IS a school project.

</div>

<div class="cell code" execution_count="51"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:29:20.096598Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:29:20.092867Z&quot;}"
collapsed="false">

``` python
bad = ['fuck', 'bitch', 'bitches', 'fucking', 'fuckin', 'shit', 'motherfucker', 'motherfuckin', 'shit', 'ass', 'asshole', 'motherfuckers', 'fucks']
# create Boolean mask based on whether word is in valid_words list
mask = df_lyricsCount['word'].isin(bad).copy()
# select rows that match the mask
df_badWords = df_lyricsCount.loc[mask]
```

</div>

<div class="cell code" execution_count="52"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:29:20.498749Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:29:20.494099Z&quot;}"
collapsed="false">

``` python
# Remove common suffix from bad words
def remove_suffix(word):
    # remove 'ing' suffix
    word = re.sub(r'ing$', '', word)
    # remove 'es' suffix
    word = re.sub(r'es', '', word)
    # remove ' ' suffix
    word = re.sub(r' ', '', word)
    # remove 'rs' suffix
    word = re.sub(r'rs', 'r', word)
    # remove 'ks' suffix
    word = re.sub(r'ks', 'k', word)
    return word

# apply function to word column
df_badWords.loc[:, 'word'] = df_badWords['word'].apply(remove_suffix)
df_sum = df_badWords.groupby('word').agg({'count': 'sum'})
# sort by count
df_sum = df_sum.sort_values('count', ascending=False).reset_index()
```

<div class="output stream stderr">

    /var/folders/pb/v7w4fnsj04d1wrdx2dfvltk00000gn/T/ipykernel_65263/917288447.py:16: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_badWords.loc[:, 'word'] = df_badWords['word'].apply(remove_suffix)

</div>

</div>

<div class="cell markdown" collapsed="false">

#### Question: Question: Using NLP to get the top 10 'Bad' Words used in Lyrics in My Spotify Playlist

</div>

<div class="cell code" execution_count="53"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:29:21.469420Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:29:21.320505Z&quot;}"
collapsed="false">

``` python
# Change labels
labels = ['F***', 'S***', 'B****', 'A**', 'M*****F*****', 'A**h***']

# Create the bar chart
plt.figure(figsize=(15, 6))
ax = sns.barplot(y='word', x='count', data=df_sum, palette="coolwarm_r")
ax.set_yticklabels(labels)
plt.title('Top 10 Bad Words used in Lyrics in My Spotify Playlist')
plt.ylabel('Word')
plt.xlabel('Frequency')
plt.show()
```

<div class="output display_data">

![](7a93108c7a0fc47a68b1b198a1edb59aaf299099.png)

</div>

</div>

<div class="cell markdown" collapsed="false">

This graph depicts the top ten bad words used in lyrics in my Spotify
playlist. Censored obviously.

</div>

<div class="cell markdown" collapsed="false">

#### This cell will be commented out because it takes a long time to run. But it creates another CSV to process for Sentiment Analysis.

</div>

<div class="cell code" collapsed="false">

``` python
"""
datafile = "/Users/mattelms/Documents/School/R-Intro-Data-Science/Spotify_Program/PySpotify/everything_lyrics.csv"
df_lyrics = pd.read_csv(datafile)
df_lyrics['sentiment'] = 0.0

# Analyze sentiment for each song
for i, row in df_lyrics.iterrows():
    lyrics = row['Lyrics']
    blob = TextBlob(lyrics, analyzer=NaiveBayesAnalyzer())
    sentiment = blob.sentiment.p_pos - blob.sentiment.p_neg
    df_lyrics.at[i, 'sentiment'] = sentiment

# Classify songs as positive or negative based on sentiment score
df_lyrics['sentiment_category'] = df_lyrics['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative')

# Print the first 10 songs and their sentiment categories
df_sent = df_lyrics[['Song', 'Artist', 'sentiment_category']]
df_sent.to_csv("sentiment.csv")
"""
```

</div>

<div class="cell markdown" collapsed="false">

#### This is the import of the CSV so I do not need to keep running the above cell.

</div>

<div class="cell code" execution_count="55"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:32:40.740517Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:32:40.722935Z&quot;}"
collapsed="false">

``` python
datafile = "/Users/mattelms/Documents/School/R-Intro-Data-Science/Spotify_Program/PySpotify/sentiments.csv"
df_sent1 = pd.read_csv(datafile)
df_sent1
```

<div class="output execute_result" execution_count="55">

                   Song         Artist sentiment_category
    0     100 Grandkids     Mac Miller           positive
    1             10:35         Tiësto           positive
    2         1, 2 Many     Luke Combs           positive
    3              1901        Phoenix           positive
    4              2055  Sleepy Hallow           positive
    ..              ...            ...                ...
    858        Daylight  David Kushner           positive
    859          Violet   Connor Price           positive
    860         Spinnin   Connor Price           negative
    861           Buddy   Connor Price           positive
    862  Jordan Belfort     Wes Walker           negative

    [863 rows x 3 columns]

</div>

</div>

<div class="cell code" execution_count="56"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:33:03.259921Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:33:03.233817Z&quot;}"
collapsed="false">

``` python
attSentiments = pd.merge(df_sent1, df_clean, on=['Song','Artist'])
attSentiments
```

<div class="output execute_result" execution_count="56">

                    Song         Artist sentiment_category        Genre  \
    0      100 Grandkids     Mac Miller           positive      Hip hop   
    1              10:35         Tiësto           positive          EDM   
    2          1, 2 Many     Luke Combs           positive      Country   
    3               1901        Phoenix           positive  Alternative   
    4               2055  Sleepy Hallow           positive          Rap   
    ..               ...            ...                ...          ...   
    858  I Ain't Worried    OneRepublic           positive         Rock   
    859      golden hour           JVKE           positive  Alternative   
    860         Daylight  David Kushner           positive          Rap   
    861            Buddy   Connor Price           positive          Rap   
    862   Jordan Belfort     Wes Walker           negative          Rap   

                                                     Album Release_Date  \
    0                                             GO:OD AM   2015-09-18   
    1                                                10:35   2022-11-03   
    2                         What You See Is What You Get   2019-11-08   
    3                             Wolfgang Amadeus Phoenix   2009-05-25   
    4                                         Still Sleep?   2021-06-02   
    ..                                                 ...          ...   
    858  I Ain’t Worried (Music From The Motion Picture...   2022-05-13   
    859            this is what ____ feels like (Vol. 1-4)   2022-09-23   
    860                                           Daylight   2023-04-14   
    861                                              Buddy   2022-12-30   
    862                                     Jordan Belfort   2015-08-26   

         Popularity  Danceability  Energy  Key  Loudness  Speechiness  \
    0            65         0.735   0.749   10  0.811862       0.0874   
    1            89         0.696   0.793    8  0.700227       0.0970   
    2            73         0.540   0.821    6  0.810556       0.0873   
    3             6         0.591   0.831    0  0.705108       0.0415   
    4            82         0.829   0.512    6  0.692736       0.1870   
    ..          ...           ...     ...  ...       ...          ...   
    858          93         0.704   0.797    0  0.689217       0.0475   
    859          90         0.515   0.593    4  0.753121       0.0322   
    860          94         0.508   0.430    2  0.487855       0.0335   
    861          69         0.914   0.506    6  0.720658       0.2230   
    862          65         0.860   0.719   11  0.780136       0.3300   

         Acousticness  Instrumentalness  Liveness  Valence     Tempo  
    0          0.3710          0.000000    0.4700    0.373  0.280841  
    1          0.0683          0.000004    0.1800    0.698  0.444883  
    2          0.0397          0.000000    0.4230    0.685  0.624590  
    3          0.0605          0.000047    0.1900    0.705  0.595171  
    4          0.4920          0.000000    0.1200    0.638  0.198417  
    ..            ...               ...       ...      ...       ...  
    858        0.0826          0.000745    0.0546    0.825  0.569645  
    859        0.6530          0.162000    0.2500    0.153  0.285515  
    860        0.8300          0.000441    0.0930    0.324  0.507835  
    861        0.2040          0.000000    0.1030    0.428  0.569777  
    862        0.1250          0.000000    0.0799    0.496  0.694807  

    [863 rows x 17 columns]

</div>

</div>

<div class="cell markdown" collapsed="false">

#### Question: Question: Using NLP and the spotify song attributes, "Valence and Energy", determine if their is a correlation between song lyric sentiment and song attributes described as determining if a songs beat is positive or negative.

</div>

<div class="cell code" execution_count="58"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:35:09.592910Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:35:09.429698Z&quot;}"
collapsed="false">

``` python
# Create a scatterplot with Sentiment variable determining color
sns.scatterplot(x='Energy', y='Valence', hue='sentiment_category', data=attSentiments, palette=['blue', 'red'])

# Add axis labels and title
plt.xlabel('Energy')
plt.ylabel('Valence')
plt.title('Valence vs Energy, Sentiment by Positive or Negative')

plt.show()
```

<div class="output display_data">

![](9d86b442fc48ae5e335717da08de82dbd2d07bda.png)

</div>

</div>

<div class="cell markdown" collapsed="false">

Valence and Energy are the most common attributes according to the
Spotify API that define how positive (good feeling) or negative (bad
feeling) the song is. I wanted to compare the valence and energy of
positive and negative songs according to the NB analyzer. However, I am
not sure if there was a correlation between them. I'm glad there was not
a correlation because valence and energy describe the beat and tone of
the song, whereas NLP is analyzing the lyrics of the song alone.

</div>

<div class="cell markdown" collapsed="false">

#### Question: Uing NLP, how do sentiments of the song lyrics correlate with song attributes?

</div>

<div class="cell code" execution_count="59"
ExecuteTime="{&quot;end_time&quot;:&quot;2023-05-07T20:37:15.151545Z&quot;,&quot;start_time&quot;:&quot;2023-05-07T20:37:14.988022Z&quot;}"
collapsed="false">

``` python
correlation = attSentiments.loc[:, ['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'sentiment_category']]
correlation['pos_neg'] = correlation['sentiment_category'].apply(lambda x: 1 if x == 'positive' else 0)

X = correlation.drop(['sentiment_category', 'pos_neg'], axis=1)
y = correlation['pos_neg']
X = sm.add_constant(X)

# Fit a logistic regression model
model = sm.Logit(y, X).fit()

# Get the coefficients and their confidence intervals
coef_df = model.params.to_frame('coef')
coef_df['lower'] = model.conf_int()[0]
coef_df['upper'] = model.conf_int()[1]

# Sort the coefficients
coef_df = coef_df.iloc[1:].sort_values('coef', ascending=False)

# Create a bar chart of the coefficients and their confidence intervals
sns.set_style('whitegrid')
sns.barplot(x='coef', y=coef_df.index, data=coef_df, palette='Blues_d')
plt.errorbar(x=coef_df['coef'], y=coef_df.index, xerr=[coef_df['coef'] - coef_df['lower'], coef_df['upper'] - coef_df['coef']], fmt='none', capsize=5, color='black')
plt.xlabel('Coefficient')
plt.ylabel('Audio Feature')
plt.title('Significant Predictors of Positive vs Negative Songs')
plt.show()
```

<div class="output stream stdout">

    Optimization terminated successfully.
             Current function value: 0.569990
             Iterations 5

</div>

<div class="output display_data">

![](fc88f37f5e720147142ffddeda74cb8b7c51982c.png)

</div>

</div>

<div class="cell markdown" collapsed="false">

This graph depicts the significant predictors of positive and negative
songs according to the NB analyzer when correlated with the attributes
provided by Spotify. This shows loudness has a higher correlation to
Positive songs and speechiness has a higher correlation to Negative
songs.

</div>
