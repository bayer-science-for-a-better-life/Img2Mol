import io
import os
import random
from base64 import encodebytes
from pathlib import Path
from urllib.parse import urlparse

IMG2MOL_DEFAULT_CUDA_VISIBLE_DEVICES = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = IMG2MOL_DEFAULT_CUDA_VISIBLE_DEVICES

import numpy as np
import torch
import torchvision
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from indigo import Indigo
from indigo.renderer import IndigoRenderer
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger

import img2mol as i2m
import utils as ut


# --- DEFAULTS
IMG2MOL_DEFAULT_MODEL_PATH = os.fspath(Path(__file__).parent.joinpath('model', 'model.ckpt'))
# Env settings from kubernetes
_p = urlparse(os.getenv("SERVICE_CDDD_BACKEND_PORT", "http://127.0.0.1:8892"))
CDDD_SERVER_HOST = _p.hostname
CDDD_SERVER_PORT = int(_p.port)


# --- ENVIRONMENT SETUP
RDLogger.DisableLog('rdApp.*')
CDDDR = ut.CDDDRequest(host=CDDD_SERVER_HOST, port=CDDD_SERVER_PORT)
model = i2m.pic2Smiles.load_from_checkpoint(IMG2MOL_DEFAULT_MODEL_PATH)
if torch.cuda.is_available():
    model.cuda()
model.eval()


# --- IMAGE PROCESSING FUNCTIONS --------------------------------------

def fitImage(im):
    old_size = im.size  # old_size[0] is in (width, height) format
    desired_size = 224

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = im.resize(new_size, Image.BICUBIC)  # Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_im = Image.new("L", (desired_size, desired_size), "white")
    new_im.paste(im, ((desired_size - new_size[0]) // 2,
                      (desired_size - new_size[1]) // 2))

    # new_im = ImageOps.expand(new_im, 15 , "white") # int(np.random.randint(15,30, size=1))
    new_im = ImageOps.expand(new_im, int(np.random.randint(5, 25, size=1)), "white")
    return new_im


def read_imagefile(file) -> Image.Image:
    img = Image.open(io.BytesIO(file))  # .convert('L')
    if img.mode == "RGBA":
        bg = Image.new('RGB', img.size, (255, 255, 255))
        # Paste image to background image
        bg.paste(img, (0, 0), img)
        return bg.convert('L')
    else:
        return img.convert('L')


def transform_Image(image):
    image = fitImage(image)
    img_PIL = torchvision.transforms.RandomRotation((-15, 15), resample=3, expand=True, center=None, fill=255)(image)
    img_PIL = torchvision.transforms.ColorJitter(brightness=[0.75, 2.0], contrast=0, saturation=0, hue=0)(img_PIL)
    shear_value = np.random.uniform(0.1, 7.0)
    shear = random.choice([[0, 0, -shear_value, shear_value], [-shear_value, shear_value, 0, 0],
                           [-shear_value, shear_value, -shear_value, shear_value]])
    img_PIL = torchvision.transforms.RandomAffine(0, translate=None, scale=None, shear=shear, resample=3,
                                                  fillcolor=255)(img_PIL)
    img_PIL = ImageEnhance.Contrast(ImageOps.autocontrast(img_PIL)).enhance(2.0)
    img_PIL = torchvision.transforms.Resize((224, 224), interpolation=3)(img_PIL)
    img_PIL = ImageOps.autocontrast(img_PIL)
    img_PIL = torchvision.transforms.ToTensor()(img_PIL)
    return img_PIL


def renderIndigo(mol, px):
    indigo = Indigo()
    renderer = IndigoRenderer(indigo)
    indigo.setOption("ignore-stereochemistry-errors", 'true')
    mol = indigo.loadMolecule(Chem.MolToSmiles(mol, isomericSmiles=False))
    mol.layout()  # if not called, will be done automatically by the renderer
    indigo.setOption("render-coloring", 'false')
    indigo.setOption("render-stereo-style", 'none')
    indigo.setOption("render-output-format", "png")
    indigo.setOption("render-superatom-mode", "collapse")
    indigo.setOption("render-image-size", px, px)
    indigo.setOption("render-background-color", 1.0, 1.0, 1.0)
    return Image.open(io.BytesIO(renderer.renderToBuffer(mol))).convert('L')


# --- IMG2MOL SERVER --------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Welcome from the Img2Mol API"}


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png", "gif")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    images = torch.cat([torch.unsqueeze(transform_Image(image), 0) for i in range(50)], dim=0).cuda()
    with torch.no_grad():
        val_cddd = model(images).detach().cpu().numpy()
    cddd = np.median(val_cddd, axis=0)
    smiles = CDDDR.cddd_to_smiles(cddd.tolist())
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol:
        can_smiles = Chem.MolToSmiles(mol)
        buf = io.BytesIO()
        img = Draw.MolToImage(mol,size=(256,256)).convert('L')
        img.save(buf, format='JPEG', quality=100)
        buf.seek(0)
        buf2 = io.BytesIO()
        image.save(buf2, format='JPEG', quality=100)
        buf2.seek(0)

        # JSON
        encoded_img = encodebytes(buf.getvalue()).decode('ascii')
        encoded_org = encodebytes(buf2.getvalue()).decode('ascii')
        return JSONResponse(
            content={'Status': 'Success', 'can_smiles': can_smiles, 'name_org': encoded_org, 'name_pred': encoded_img})

    else:
        buf = io.BytesIO()
        img = Image.open("failed.png")
        img.save(buf, format='PNG', quality=100)
        buf.seek(0)

        buf2 = io.BytesIO()
        image.save(buf2, format='JPEG', quality=100)
        buf2.seek(0)
        encoded_org = encodebytes(buf2.getvalue()).decode('ascii')
        failure = encodebytes(buf.getvalue()).decode('ascii')
        can_smiles = "No valid SMILES recognized"
        return JSONResponse(
            content={'Status': 'Failure', 'can_smiles': can_smiles, 'name_org': encoded_org, 'name_pred': failure})


if __name__ == "__main__":
    import uvicorn

    # local server for debugging
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8580,
        log_level="info",
        workers=1,
        debug=False,
    )
