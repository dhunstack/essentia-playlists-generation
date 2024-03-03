# Automated Playlist Generation

Used MusAV dataset for playlist generation with audio features and embeddings.

## Quick Start

### Feature Extraction

#### Setup

The following parameters need to be set in the `config.py` file -

- `DATA_PATH` - Root Path of the folder with all audio files. Current configuration supports MP3 files with 44,100 Hz sampling rate.

MusAV dataset access can be requested [here](https://zenodo.org/records/7448344).

### Playlist Generators

#### Usage

Run the audio descriptors playlist app using -

``` python
streamlit run playlist_descriptor.py
```

Run the track similarity playlist app using -

``` python
streamlit run playlist_embeddings.py
```

## Documentation

<!-- A report (~2 pages) describing the decisions you took in all steps, when generating the features,  computing statistics overview, building the interface, along with your personal opinion of the quality of the system in terms of its capability to generate playlists. Include your observations on the quality of the extracted features, including examples of good and bad extracted features that you encountered. -->

### Audio Content based Playlists

#### Introduction

In this project, I've created scripts for analyzing and previewing music in any given collection -

- `main.py` - Extract audio features and embeddings of 44.1kHz MP3 files from any custom music collection
- `collection_overview.ipynb` - Compute analysis statistics for the music collection
- `playlist_descriptor.py` - Create music playlists filtered by descriptors such as key, tempo, music style, danceability, voice vs instrumental, and arousal-valence
- `playlist_embeddings.py` - Generate music playlists based on queries by track example, finding tracks similar to a given query track.

---

#### Audio Analysis with Essentia

**Objective:** Developed a standalone Python script to analyze an entire audio collection (MusAV for this assignment) and extract specific descriptors for each track using Essentia.

**Descriptors Extraction:**

- Tempo (BPM) - Used `TempoCNN` with `deepsquare-k16` square filters. Based on the [referenced paper](https://arxiv.org/abs/1903.10839), deepsquare performs best for tempo detection on 3 out of 4 datasets. Additional computation cost is okay since we store results.
- Key - `KeyExtractor` with `temperley`, `krumhansl`, `edma` profiles
- Loudness - `LoudnessEBUR128` for integrated loudness in LUFS
- Embeddings - `Discogs-Effnet` and `MSD-MusiCNN` models
- Music styles - `Discogs-Effnet` with activations for 400 music styles
- Voice/instrumental classifier - Model based on `Discogs-Effnet` embeddings
- Danceability - Classifier model with `Discogs-Effnet` embeddings
- Arousal and valence (music emotion) - emoMusic pretrained model with `MSD-MusiCNN` embeddings

**Loading Audio:**

- Load audio with `AudioLoader` - 44.1 kHz stereo for Loudness
- Downmix to mon with `MonoMixer` -  44.1 kHz mono for Key
- `Resample` 44.1 kHz mono to 11,025 Hz mono for TempoCNN
- `Resample` 44.1 kHz mono to 16kHz mono for Embeddings

**Using ML Models:**

- Instantiated necessary algorithms once.
- Computed required embeddings once for each track.

**Script Design:**

- Script runnable on any music collection of 44.1kHz MP3s with any size and nested folder structure.
- `DATA_PATH` can be set in `config.py` file
- Five separate `pickle` files in `features` folder for storing - analysis results, two sets of embeddings, genre activations and file paths
- Error handling implemented to skip files with analysis errors
- Added a `tqdm` progress bar

---

#### Music Collection Overview

**Objective:** Generate a statistical report from analyzed audio files to explore the music collection.

**Report Commentary:**

- Single music style chosen per track, the one with the maximum activation
- Comment on the collection's diversity in terms of music styles, tempos, tonality, emotion, etc. -
  - Music Style - Classes are heavily unbalanced in favour of popular styles Rock, Electronic, Hip Hop. But it seems reasonable based on the average consumption pattern of the population.
  - Tempos - Seems to have 2 peaks, one around 90 and another around 125. Distribution seems to reflect popular music patterns.
  - Tonality - C, G, Am, F seem to be the most popular keys in Krumhansl, again reflecting popular music.
  - Emotion - Positive valence and high arousal values, as expected with popular music.
  
- Comment on differences in key and scale estimation using the three profiles (`temperley`, `krumhansl`, `edma`). What is the % of tracks on which all three estimations agree? - All three estimations agree on around `49%` of tracks
- If we had to select only one profile to present to the users, which one should we use? - `krumhansl` based on the analysis in my statistical report
- Comment on loudness. Does its distribution make sense? - Yeah it seems reasonable, detailed explanation in the statistical report.

---

#### Playlists Based on Descriptor Queries

**Objective:** Create a simple UI for playlist generation based on audio analysis results using Streamlit.

- Extend the provided Streamlit code to implement search by various analysis results such as tempo, voice/instrumental classification, danceability, arousal/valence, and key/scale.
  - Track is picked if the activation in any selected style is within the range, unlike original code where activation was checked in all selected styles.
  - Range selection filters for tempo, danceability, arousal and valence
  - Multiselect for key, scale and vocals, to allow maximum flexibility

- `krumhansl` key/scale estimation profile picked for UI
  
**Observations and Issues:**

- Style - Activation ranges had to be set wide to get one song for some styles. Was helpful to have the style summary in picking threshold.
- Tempo - Worked well
- Voice Instrumental - distribution has some songs in the middle, not towards either extremities. Precision recall tradeoff in picking activation threshold that performs well.
- Danceability - skewed distribution, too many values close to 1, didn't seem to be meaningful
- Arousal Valence - Found the predictions more reasonable
- Key - Worked well

One extension could be to add category sorting too, instead of just filtering.

---

#### Playlists Based on Track Similarity

**Objective:**

- Allow users to select a query track and generate two lists of the 10 most similar tracks using effnet-discogs and msd-musicnn embeddings.
- Compute music similarity using cosine distance between the query and other tracks' embeddings.
- Embed music players to listen and compare results.

**Observations:**

- Discogs Embedding seemed to be better at giving results from the same genre and with similar style.
- MSD Embedding seemed to be capturing certain elements of the song, and would give results that had some similarity but not necessarily from the same genre.
- For certain styles, both the embeddings produced similar results.
- I would pick Discogs Embedding for building a similarity ranking, it seemed more robust.
- For a recommender system, if I wanted it to have more surprise element, I could probably use MSD Embeddings.
  
---
  