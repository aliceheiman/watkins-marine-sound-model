import streamlit as st
from fastcore.all import *
from fastai.vision.all import *
import os
from scipy.io import wavfile
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

########## SETUP
model_path = Path(os.path.join("models", "export.pkl"))
asset_path = Path("assets")
audio_path = Path("audio")
thumb_path = Path("thumbnails")
learn = load_learner(model_path)


def classify_img(data):
    pred, pred_idx, probs = learn.predict(data)
    return pred, probs[pred_idx]


def classify_wav(data, fname=False):
    if fname == False:
        with open(asset_path / "temp.wav", "wb") as f:
            f.write(data)

        sample_rate, samples = wavfile.read(asset_path / "temp.wav")
    else:
        sample_rate, samples = wavfile.read(data)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.grid(False)
    ax.specgram(samples, Fs=sample_rate)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(asset_path / "temp.jpg")

    with open(asset_path / "temp.jpg", "rb") as file:
        image_bytes = file.read()

    return classify_img(image_bytes)


########## WEBPAGE

st.title("🐬 Under the Sea - Deep Learning for Marine Mammal Sound Classification")

st.image("assets/lineup.png")

st.markdown("Jump directly to [Demo](#demo)")

st.subheader("Introduction")
st.write(
    "- Machine Learning techniques have great potential to **analyze complex maritime data from various sensors**, such as detecting species of fish from videos or observe habitat changes over time through sensors."
)

st.write(
    "- One application of machine learning in marine biology is to use Convolutional Neural Networks (CNNs) to **detect and count marine animals in an image or video**."
)

st.write(
    "- Even **audio recordings** can be analyzed in a similar fashion. It is possible to convert other mediums into their **image representation** to leverage the high utility of CNNs. For example, an audio sample can be converted into a **spectrogram** that is then fed to the model."
)

st.write(
    "**Objective:** In this project, I use Deep Learning (DL) to develop a model capable of classifying marine animals based on audio recordings."
)

st.subheader("Method")

st.write(
    "I used a subset of the 'Best of' Audio files from the Watkins Marine Mammal Sound Database. I chose a subset of five different species from the whole database for this project. Thus, the five categories are Narwhal, Walrus, Harp Seal, Spinner Dolphin, and Killer Whale. I used a Python script to download 30 to 40 samples of audio recordings for each category."
)

st.write("Below, you can hear some examples of how the different species sound!")

st.write("**Narwhal:**")
st.audio(os.path.join("audio", "Narwhal", "6800200A.wav"))
st.image(os.path.join("thumbnails", "narwhal.png"))

st.write("**Walrus:**")
st.audio(os.path.join("audio", "Walrus", "7200200C.wav"))
st.image(os.path.join("thumbnails", "walrus.png"))

st.write("**Harp Seal:**")
st.audio(os.path.join("audio", "Harp Seal", "6700900D.wav"))
st.image(os.path.join("thumbnails", "harp-seal.png"))

st.write("**Spinner Dolphin:**")
st.audio(os.path.join("audio", "Spinner Dolphin", "7100100N.wav"))
st.image(os.path.join("thumbnails", "spinner-dolphin.png"))

st.write("**Killer Whale:**")
st.audio(os.path.join("audio", "Killer Whale", "6002600A.wav"))
st.image(os.path.join("thumbnails", "killer-whale.png"))

st.write(
    "Using Matplotlib and Scipy, I then converted the audio files into a spectrogram, with the axis and grid overlays removed. Below, you can see some examples of how the radio recordings look in spectrogram form."
)

st.image(os.path.join("assets", "spectrograms.png"))

st.write(
    "Using the deep learning framework Fast AI, I loaded the spectrograms into a training, validation, and testing set. Then, I fine-tuned the convolutional neural network Resnet34 for 10 epochs using a learning rate of 0.00083. Finally, I exported the model and generated a confusion matrix to find the model's confusion points."
)

st.subheader("Results")

st.write(
    "With 30-40 training examples for each category and 10 epochs of fine-tuning, the model achieved 84 % accuracy."
)

df = pd.DataFrame(
    np.array(["84.2", "86.3", "85.9", "85.4"]).reshape(1, 4),
    columns=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"],
)

st.table(df)

st.write(
    "The confusion matrix below shows that the model has most trouble distinguishing between a walrus and a harp seal."
)

st.image(os.path.join("assets", "confused.png"))

st.subheader("Demo")

st.write(
    "Select your own  audio file to test the model with, or use one of the examples below."
)

uploaded_file = st.file_uploader("Choose wav file")
option = st.selectbox(
    "Demo Examples",
    (
        "Select example...",
        "Narwhal",
        "Walrus",
        "Harp Seal",
        "Spinner Dolphin",
        "Killer Whale",
    ),
)

bytes_data = None
demo_audio = None

if option:
    if option == "Narwhal":
        demo_audio = os.path.join("assets", "demo-narwhal.wav")
    if option == "Walrus":
        demo_audio = os.path.join("assets", "demo-walrus.wav")
    if option == "Harp Seal":
        demo_audio = os.path.join("assets", "demo-harp-seal.wav")
    if option == "Spinner Dolphin":
        demo_audio = os.path.join("assets", "demo-spinner-dolphin.wav")
    if option == "Killer Whale":
        demo_audio = os.path.join("assets", "demo-killer-whale.wav")

    if demo_audio:
        st.audio(demo_audio)


if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

classify = st.button("Classify")
if classify and (demo_audio or bytes_data):
    if demo_audio == None:
        label, confidence = classify_wav(bytes_data)
    else:
        label, confidence = classify_wav(demo_audio, fname=True)

    st.markdown(f"##### Prediction: {label}")
    st.write(f"Confidence: {confidence:.04f}")

    if label == "Killer Whale":
        st.markdown(
            '"""*The killer whale, also known as orca, is the ocean’s top predator. It is the largest member of the Delphinidae family, or dolphins. Members of this family include all dolphin species, as well as other larger species, such as long-finned pilot whales and short-finned pilot whales, whose common names also contain "whale" instead of "dolphin."*"""'
        )
        st.caption(
            "Information from :blue[NOAA Fisheries]: https://www.fisheries.noaa.gov/species/killer-whale"
        )
        st.write("Video of Killer Whales from BBC Earth:")
        st.video("https://www.youtube.com/watch?v=fs8ZveNZQ8g&ab_channel=BBCEarth")

    if label == "Harp Seal":
        st.markdown(
            '"""*Harp seals live throughout the cold waters of the North Atlantic and Arctic Oceans. Three populations in the Barents Sea, East Coast of Greenland, and Northwest Atlantic Ocean are recognized based on geographic distribution as well as morphological, genetic, and behavioral differences. These seals are named after the black patch on their back, which looks like a harp.*"""'
        )
        st.caption(
            "Information from :blue[NOAA Fisheries]: https://www.fisheries.noaa.gov/species/harp-seal"
        )
        st.write("Video of Harp Seals from National Geographic:")
        st.video("https://youtu.be/BF2TZq-ntRQ?si=4rLHEynd6CR_ltfq&t=501")

    if label == "Walrus":
        st.markdown(
            '"""*The Latin name for the walrus translates as “tooth-walking sea horse.” You can understand why.  Walruses use their long ivory tusks to haul their heavy bodies up onto the ice, to forage for food, and to defend against predators.*"""'
        )
        st.caption(
            "Information from :blue[NOAA Fisheries]: https://oceantoday.noaa.gov/animalsoftheice_walruses/"
        )
        st.write("Video of the Walrus from the Netflix series 'Our Planet':")
        st.video("https://www.youtube.com/watch?v=qVJzQc9ELTE&ab_channel=Netflix")

    if label == "Spinner Dolphin":
        st.markdown(
            '"""*Spinner dolphins are probably the most frequently encountered cetacean in nearshore waters of the Pacific Islands Region. Spinner dolphins received their common name because they are often seen leaping and spinning out of the water. The species\' name, longirostris, is Latin for “long beak,” referring to their slender shaped beak or rostrum.*"""'
        )
        st.caption(
            "Information from :blue[NOAA Fisheries]: https://www.fisheries.noaa.gov/species/spinner-dolphin"
        )
        st.write("Video of Spinner Dolphins from the Netflix series 'Our Planet':")
        st.video("https://www.youtube.com/watch?v=2vwY1bN6bx8&ab_channel=Netflix")

    if label == "Narwhal":
        st.markdown(
            '"""*Narwhals are found in the Arctic Ocean. Generally male narwhals have a tooth that grows into a long clockwise-spiraled tusk, resembling a unicorn horn.*"""'
        )
        st.caption(
            "Information from :blue[NOAA Fisheries]: https://www.fisheries.noaa.gov/species/narwhal"
        )
        st.write("Video of Narwhals from the Netflix series 'Our Planet':")
        st.video("https://www.youtube.com/watch?v=UVwYygnGkPE&ab_channel=Netflix")

st.subheader("Discussion")

st.write(
    "This project gave me the opportunity to explore an application of machine learning in marine biology. There has been extensive research in species identification using audio and machine learning, but future directions include identifying individuals within a species."
)

st.write(
    "In the process, I faced some challenges in working with the audio of marine animals."
)

st.write(
    "- It is difficult to know exactly what in the audio files the machine learning model is using to make its classifications. For example, the ambient noise, the microphone used, and other acoustic effects can have a large impact on the predictions, making it less likely that the model will generalize to new samples taken at a different time with different audio equipment. "
)

st.write(
    "- The audio quality of the samples varies greatly, with some samples containing very little of the marine animal and more of the environment."
)

st.write(
    "- The audio samples vary in length, so the spectrograms are not standardized. "
)

st.write(
    "The main things I learned from this project were how to use transfer learning to fine-tune a pre-trained model, and a little insight into some of the challenges of working with audio files and in the field of marine biology!"
)

st.write("Thanks for reading! :)")

st.subheader("References")

st.write(
    "Goodwin, M., Halvorsen, K. T., Jiao, L., Knausgård, K. M., Martin, A. H., Moyano, M., Oomen, R. A., Rasmussen, J. H., Sørdalen, T. K., & Thorbjørnsen, S. H. (2022). Unlocking the potential of deep learning for marine ecology: Overview, applications, and outlook†. ICES Journal of Marine Science, 79(2), 319–336. https://doi.org/10.1093/icesjms/fsab255"
)

st.write(
    "Huang, H. C., Joseph, J., Huang, M. J., & Margolina, T. (2016). Automated detection and identification of blue and fin whale foraging calls by combining pattern recognition and machine learning techniques. OCEANS 2016 MTS/IEEE Monterey, 1–7. https://doi.org/10.1109/OCEANS.2016.7761269"
)

st.write(
    "NOAA Fisheries. Retrieved January 1, 2024, from https://www.fisheries.noaa.gov/"
)

st.write(
    "Stowell, D. (2022). Computational bioacoustics with deep learning: A review and roadmap. PeerJ, 10, e13152. https://doi.org/10.7717/peerj.13152"
)

st.write(
    "von Schuckmann, K., Minière, A., Gues, F., Cuesta-Valero, F. J., Kirchengast, G., Adusumilli, S., Straneo, F., Ablain, M., Allan, R. P., Barker, P. M., Beltrami, H., Blazquez, A., Boyer, T., Cheng, L., Church, J., Desbruyeres, D., Dolman, H., Domingues, C. M., García-García, A., … Zemp, M. (2023). Heat stored in the Earth system 1960–2020: Where does the energy go? Earth System Science Data, 15(4), 1675–1709. https://doi.org/10.5194/essd-15-1675-2023"
)
