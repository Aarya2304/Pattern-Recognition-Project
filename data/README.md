# Dataset

The experiment uses the official **UCI Human Activity Recognition Using Smartphones** dataset (UCI dataset 240): https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones. The loader downloads the official archive when possible. The archive supplied by UCI contains a nested `UCI HAR Dataset.zip`, which the loader extracts and validates.

For offline setup, place the extracted `UCI HAR Dataset/` directory or the official archive in `data/raw/`. Expected files include `features.txt`, `activity_labels.txt`, `train/`, `test/`, and the nine per-axis files under each `Inertial Signals/` directory. `python main.py --mode validate` reports dimensions, label alignment, participant sets, and validity checks.

The original data are attributed to Reyes-Ortiz, Anguita, Ghio and Oneto; see the dataset README and dataset page for citation and license. Raw/processed data are excluded from source control. No sample or synthetic dataset is used for final experiments.
