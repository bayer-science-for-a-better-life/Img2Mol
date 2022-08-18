import base64
import os
from io import BytesIO
from urllib.parse import urlparse

import requests
import streamlit as st
from PIL import Image

# Env settings from kubernetes
_p = urlparse(os.getenv("SERVICE_IMG2MOL_BACKEND_PORT", "http://127.0.0.1:8580"))
IMG2MOL_SERVER_HOST = _p.hostname
IMG2MOL_SERVER_PORT = int(_p.port)


st.set_page_config(page_title="Img2Mol", page_icon="icon.png")

hide_streamlit_style = """
            <style>
            MainMenu {visibility: visible;}
            footer {visibility: hidden;}
            footer:after {
                content:'Provided by DT Machine Learning Research'; 
                visibility: visible;
            }
            </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Img2Mol - Demo Version")

st.markdown(
    """
    Img2Mol is a fast algorithm for automatic recognition of the molecular content of a molecule's graphical depiction.
    To read more about how the model works, click <a href="https://chemrxiv.org/articles/preprint/Img2Mol_-_Accurate_SMILES_Recognition_from_Molecular_Graphical_Depictions/14320907">here</a>.

    This proof of concept web application converts images containing 2D structural representations of a molecule to the corresponding SMILES representation.
    In its current state not all special cases are covered, i.e. try to avoid uploading images that containt text artifacts due to cropping.

    *Please note that the current deployment of this web application is temporary and we will provide a more permanent instance in the near future.*    
    """
    , unsafe_allow_html=True)

image_app = Image.open("Img2MolApp.png")
st.image(image_app, width=700)

# displays a file uploader widget
image = st.file_uploader("Choose an image to process")

# displays the select widget for the styles
# style = st.selectbox("Choose the style", [i for i in STYLES.keys()])

# displays a button
if st.button("Get SMILES"):
    if image is not None:
        files = {"file": image}
        res = requests.post(f"http://{IMG2MOL_SERVER_HOST}:{IMG2MOL_SERVER_PORT}/predict", files=files)
        if res.status_code == 200:
            img_path = res.json()
            image_orig = Image.open(BytesIO(base64.b64decode(img_path['name_org'])))
            image_pred = Image.open(BytesIO(base64.b64decode(img_path['name_pred'])))

            images = [image_orig, image_pred]
            st.image(images, use_column_width=False, width=340, caption=["Input image", "Molecule from predicted SMILES"])

            st.write("Predicted SMILES")
            st.write(img_path['can_smiles'])
        else:
            st.error(
                f"Received response code {res.status_code} from API. Please try with another image."
            )
