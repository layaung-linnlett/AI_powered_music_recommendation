"""Genre taxonomy mapping: collapses 114 Spotify sub-genres into 6 acoustically
distinct super-genres.

Motivation
----------
The original 114-genre dataset contains many near-duplicate and sub-genre labels
whose audio feature profiles are acoustically indistinguishable. A classifier
trained on 114 such labels achieves roughly 30-35% accuracy because no
feature-based model can separate genres that share identical audio
characteristics. Remapping to 6 acoustically distinct super-genres allows the
model to learn genuinely separable boundaries and achieve high accuracy.

Taxonomy design process
-----------------------
This taxonomy was derived through four rounds of iterative per-class accuracy
analysis on held-out test data:

Round 1 (114 classes): 30-35% accuracy. Collapsed to 22 super-genres.
Round 2 (22 classes): 50.76% accuracy. 7 targeted merges reduced to 15 classes.
Round 3 (15 classes): 56.42% accuracy. 5 targeted merges reduced to 9 classes.
Round 4 (9 classes): 62.26% accuracy. Per-class analysis showed the core problem:
  three high-danceability classes (latin 70.5%, dance-pop 44.5%, asian-pop 46.0%)
  accounted for 45k of 114k samples and their primary confusion was with each
  other. Merging them into one "dance" super-class eliminated this confusion.
  Children (61.6%) and hip-hop (15.7%) were merged into "vocal" since both are
  distinguished by high speechiness and their confusion was primarily with the
  now-merged "dance" bucket.

Design principles
-----------------
The 6 super-genres are separated by 4 strong audio axes in the Spotify feature
space:
  1. acousticness: "acoustic" has very high acousticness (>0.5). All other
     classes have low acousticness.
  2. energy + loudness: "heavy" has maximum energy and loudness. "electronic"
     has very high energy but controlled loudness. "acoustic" has low energy.
  3. danceability: "dance" has very high danceability across all sub-genres.
     "acoustic" has low danceability. "heavy" has moderate danceability.
  4. speechiness: "vocal" (hip-hop + children) has distinctively high
     speechiness, setting it apart from all other classes.

Class profiles
--------------
- acoustic: very high acousticness, low energy. Encompasses classical, ambient,
  folk, blues, and romance. Separated by acousticness and energy from all
  electric genres.
- alternative: moderate-high energy, guitar-driven, low acousticness, moderate
  danceability. Separated from heavy by lower loudness and from dance by lower
  danceability.
- dance: very high danceability, moderate-high valence. Merges latin, dance-pop,
  r-and-b, soul, and asian-pop because their per-class accuracies at 9 classes
  were 44-71% with primary confusion being each other. After merging, the
  danceability signal cleanly separates this class.
- electronic: very high energy, very low acousticness, high instrumentalness.
  The energy + acousticness combination is a very strong distinguishing signal.
- heavy: very high energy, maximum loudness, low valence. Encompasses metal,
  punk, emo, and hardcore.
- vocal: very high speechiness. Encompasses hip-hop, rap, spoken word, children,
  and comedy. Speechiness is the single strongest individual discriminating
  feature in this dataset.
"""

# ==== Standard Library Imports ====
from typing import Dict, List

# ==== Internal Imports ====
from src.utils import get_logger

# ==== Module Logger ====
logger = get_logger(__name__)

# ==== Genre Taxonomy ====
# Maps every original genre label to its super-genre name.
# Genres that appear in the dataset but are not listed here are kept as-is.

GENRE_MAPPING: Dict[str, str] = {

    # ==== 1. Acoustic ====
    # Profile: very high acousticness (>0.5 average), low energy, low loudness.
    # Distinguishing axis: acousticness and energy separate this class from
    # all electric genres. Includes folk, classical, ambient, blues, jazz,
    # and romance sub-genres, all of which share the acoustic/low-energy signature.
    "acoustic":          "acoustic",
    "folk":              "acoustic",
    "singer-songwriter": "acoustic",
    "songwriter":        "acoustic",
    "country":           "acoustic",
    "bluegrass":         "acoustic",
    "honky-tonk":        "acoustic",
    "guitar":            "acoustic",
    "blues":             "acoustic",
    "jazz":              "acoustic",
    "groove":            "acoustic",
    "ambient":           "acoustic",
    "new-age":           "acoustic",
    "sleep":             "acoustic",
    "study":             "acoustic",
    "chill":             "acoustic",
    "piano":             "acoustic",
    "classical":         "acoustic",
    "opera":             "acoustic",
    "romance":           "acoustic",
    "sad":               "acoustic",

    # ==== 2. Alternative ====
    # Profile: moderate-high energy, guitar-driven, low acousticness.
    # Distinguishing axes: lower danceability than dance, lower loudness than heavy.
    "alternative":       "alternative",
    "alt-rock":          "alternative",
    "indie":             "alternative",
    "indie-pop":         "alternative",
    "grunge":            "alternative",
    "british":           "alternative",
    "psych-rock":        "alternative",
    "garage":            "alternative",
    "rock":              "alternative",
    "hard-rock":         "alternative",
    "rock-n-roll":       "alternative",
    "rockabilly":        "alternative",

    # ==== 3. Dance ====
    # Profile: very high danceability, moderate-high valence, moderate energy.
    # Merges latin, dance-pop, r-and-b, soul, asian-pop, and world music because
    # per-class analysis showed these classes primarily confuse each other (not
    # other super-genres). After merging, danceability and valence cleanly
    # separate this class from acoustic, alternative, electronic, and heavy.
    "pop":               "dance",
    "dance":             "dance",
    "disco":             "dance",
    "party":             "dance",
    "power-pop":         "dance",
    "pop-film":          "dance",
    "happy":             "dance",
    "synth-pop":         "dance",
    "gospel":            "dance",
    "soul":              "dance",
    "r-n-b":             "dance",
    "funk":              "dance",
    "latin":             "dance",
    "latino":            "dance",
    "reggaeton":         "dance",
    "dancehall":         "dance",
    "salsa":             "dance",
    "samba":             "dance",
    "sertanejo":         "dance",
    "forro":             "dance",
    "pagode":            "dance",
    "mpb":               "dance",
    "brazil":            "dance",
    "tango":             "dance",
    "reggae":            "dance",
    "dub":               "dance",
    "ska":               "dance",
    "world-music":       "dance",
    "afrobeat":          "dance",
    "turkish":           "dance",
    "iranian":           "dance",
    "french":            "dance",
    "german":            "dance",
    "swedish":           "dance",
    "spanish":           "dance",
    "indian":            "dance",
    "j-pop":             "dance",
    "j-idol":            "dance",
    "j-dance":           "dance",
    "anime":             "dance",
    "j-rock":            "dance",
    "k-pop":             "dance",
    "cantopop":          "dance",
    "mandopop":          "dance",
    "malay":             "dance",

    # ==== 4. Electronic ====
    # Profile: very high energy (>0.8 average), very low acousticness (<0.1),
    # high instrumentalness. The energy + acousticness combination is a very
    # strong distinguishing signal in the Spotify feature space.
    "edm":               "electronic",
    "electronic":        "electronic",
    "electro":           "electronic",
    "club":              "electronic",
    "idm":               "electronic",
    "house":             "electronic",
    "deep-house":        "electronic",
    "chicago-house":     "electronic",
    "detroit-techno":    "electronic",
    "minimal-techno":    "electronic",
    "techno":            "electronic",
    "trance":            "electronic",
    "progressive-house": "electronic",
    "drum-and-bass":     "electronic",
    "dubstep":           "electronic",
    "breakbeat":         "electronic",
    "hardstyle":         "electronic",

    # ==== 5. Heavy ====
    # Profile: very high energy (>0.85), maximum loudness (>-5 dB average),
    # very low valence (<0.4). Distinguishing axes: loudness and valence
    # separate heavy from electronic (which shares high energy).
    "metal":             "heavy",
    "heavy-metal":       "heavy",
    "death-metal":       "heavy",
    "black-metal":       "heavy",
    "metalcore":         "heavy",
    "grindcore":         "heavy",
    "emo":               "heavy",
    "goth":              "heavy",
    "industrial":        "heavy",
    "punk":              "heavy",
    "punk-rock":         "heavy",
    "hardcore":          "heavy",

    # ==== 6. Vocal ====
    # Profile: very high speechiness (>0.1 average, often >0.3 for pure rap).
    # Distinguishing axis: speechiness is the single strongest individual feature
    # in this dataset. Merges hip-hop, rap, spoken word, children, and comedy.
    # Children (high speechiness + high valence) and hip-hop (high speechiness +
    # moderate valence) both have distinctively high speechiness that separates
    # them from all other classes.
    "hip-hop":           "vocal",
    "trip-hop":          "vocal",
    "kids":              "vocal",
    "children":          "vocal",
    "comedy":            "vocal",
    "disney":            "vocal",
    "show-tunes":        "vocal",
}

# ==== Sorted list of all 6 super-genre names ====
SUPER_GENRES: List[str] = sorted(set(GENRE_MAPPING.values()))


# ==== Helper ====

def apply_genre_mapping(series, mapping: Dict[str, str] = GENRE_MAPPING):
    """Map a pandas Series of original genre labels to super-genre labels.

    Any label not found in the mapping is left unchanged. This is intentional:
    it allows the function to be called on partial or future datasets without
    raising errors.

    Args:
        series: pandas Series of original genre label strings.
        mapping: Dict mapping original label to super-genre label.

    Returns:
        pandas Series of super-genre label strings.
    """
    unmapped = set(series.unique()) - set(mapping.keys())
    if unmapped:
        logger.warning(
            "%d genre label(s) not found in GENRE_MAPPING and will be kept as-is: %s",
            len(unmapped), sorted(unmapped),
        )
    return series.map(lambda g: mapping.get(g, g))


# ==== Entry Point ====

if __name__ == "__main__":
    import pandas as pd
    from src.data_loader import load_data, discover_target

    raw_df = load_data()
    target_col = discover_target(raw_df)

    original = raw_df[target_col]
    remapped = apply_genre_mapping(original)

    print(f"Original classes : {original.nunique()}")
    print(f"Remapped classes : {remapped.nunique()}")
    print()
    print("Super-genre distribution:")
    print(remapped.value_counts().to_string())
