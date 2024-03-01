# Automated Playlist Generation

Used MusAV dataset for playlist generation with audio features and embeddings.

## Quick Start

### Feature Extraction

#### Setup

The following parameters need to be set in the `config.py` file -

- `DATA_PATH` - Root Path of the folder with all audio files. Current configuration supports MP3 files with 44,100 Hz sampling rate.

MusAV dataset access can be requested [here](https://zenodo.org/records/7448344).

#### Usage

Extract features and embedding with the following command -

``` python
python main.py
```

## Documentation

This project has three major components -

- Feature Extraction with [Essentia](https://essentia.upf.edu/index.html)
- Statistical Overview of the Dataset
- Playlist Generation

### Feature Extraction with Essentia

The following features are extracted from audio files

- [Tempo(BPM)](https://essentia.upf.edu/models.html#tempocnn)
- [Key](https://essentia.upf.edu/reference/std_KeyExtractor.html)
- [Loudness](https://essentia.upf.edu/reference/std_LoudnessEBUR128.html)
- Embeddings - [Discogs](https://essentia.upf.edu/models.html#discogs-effnet) and [MSD](https://essentia.upf.edu/models.html#msd-musicnn) from Essentia Models
- [Genre](https://essentia.upf.edu/models.html#genre-discogs400)
- [Instrumental Probability](https://essentia.upf.edu/models.html#voice-instrumental)
- [Danceability](https://essentia.upf.edu/models.html#danceability)
- [Arousal](https://essentia.upf.edu/models.html#arousal-valence-emomusic)
- [Valence](https://essentia.upf.edu/models.html#arousal-valence-emomusic)
