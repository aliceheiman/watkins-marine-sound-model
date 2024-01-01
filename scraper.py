"""
This script downloads the `Best of` Audio files from the Watkins Marine Mammal Sound Database.

Credit: Watkins Marine Mammal Sound Database, Woods Hole Oceanographic Institution and the New Bedford Whaling Museum
URL: https://whoicf2.whoi.edu/science/B/whalesounds/index.cfm

Date of script: 30 December, 2023 
"""

import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

from scipy import signal
from scipy.io import wavfile

BASE_URL = "https://whoicf2.whoi.edu"
SUBSET = ["Killer Whale", "Spinner Dolphin", "Harp Seal", "Walrus", "Narwhal"]

audio_path = Path("audio")
thumbnail_path = Path("thumbnails")
spectrograms_path = Path("spectrograms")

audio_path.mkdir(exist_ok=True)
thumbnail_path.mkdir(exist_ok=True)
spectrograms_path.mkdir(exist_ok=True)


def download_audio_from(page_url, foldername, n=20):
    """Downloads all .wav files from a `Best of` page from the Watkins Marine Sound Database.

    Args:
        page_url (string): Watkins Best of Page with audio download links
        foldername (string): Name of species
    """
    print(f"Fetching {page_url}")
    page = requests.get(page_url)
    page.raise_for_status()

    folder_path = audio_path / foldername
    folder_path.mkdir(exist_ok=True)

    soup = BeautifulSoup(page.content, "html.parser")

    links = soup.find_all("a")
    download_links = [link for link in links if link.text == "Download"]
    pattern = re.compile("(\w*.wav)$")

    total_saved = 0
    for link in tqdm(download_links[:n]):
        filename = pattern.findall(link["href"])[0]

        if filename.endswith(".wav"):
            audiofile = requests.get(BASE_URL + link["href"])
            with open(folder_path / filename, "wb") as f:
                f.write(audiofile.content)
                total_saved += 1

    print(f"Saved {total_saved} audio files to {folder_path}")


def download_thumbnail_from(image_url, name):
    """Downloads a PNG thumbnail from a given url.

    Args:
        image_url (string): URL to Marine Mammal image PNG.
        name (string): Name of the animal depicted in the image.
    """

    print(f"Downloading thumbnail from {image_url}")
    filename = name.lower().replace(" ", "-") + ".png"

    thumbnail = requests.get(BASE_URL + image_url)
    with open(thumbnail_path / filename, "wb") as f:
        f.write(thumbnail.content)

    print(f"Saved thumbnail to {filename}")


def generate_spectrogram(foldername):
    folder_path = spectrograms_path / foldername
    folder_path.mkdir(exist_ok=True)

    for filename in os.listdir(audio_path / foldername):
        audio_filepath = audio_path / foldername / filename
        output_filepath = folder_path / (filename[:-4] + ".jpg")

        sample_rate, samples = wavfile.read(audio_filepath)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        ax.grid(False)
        ax.specgram(samples, Fs=sample_rate)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_filepath)

    print(f"Saved spectrograms to {folder_path}.")


def get_best_of():
    print("Fetching species...")
    url = "https://whoicf2.whoi.edu/science/B/whalesounds/index.cfm"
    page = requests.get(url)
    page.raise_for_status()

    soup = BeautifulSoup(page.content, "html.parser")
    links = soup.find_all("a")

    best_of = [link for link in links if link["href"].startswith("bestOf")]
    data = []

    print("Getting species data...")
    for link in tqdm(best_of):
        page_name = link.find_all("h3")[0].text
        page_name = (
            page_name.replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .replace("'", "")
        )
        page_href = f"https://whoicf2.whoi.edu/science/B/whalesounds/{link['href']}"
        image_src = link.find_all("img")[0]["src"]

        if SUBSET == None or page_name in SUBSET:
            data.append((page_name, page_href, image_src))

    return data


if __name__ == "__main__":
    watkins = get_best_of()

    for species_data in watkins:
        name, url, image_src = species_data

        download_thumbnail_from(image_src, name)
        download_audio_from(url, name, n=40)
        generate_spectrogram(name)
        print()
