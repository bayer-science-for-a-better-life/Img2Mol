import os
import numpy as np
import logging
import argparse
import json
from flask import Flask, jsonify, request
from cddd.inference import InferenceServer
from rdkit import Chem
from cddd.preprocessing import remove_salt_stereo, organic_filter, REMOVER


logging.getLogger('tensorflow').disabled = True
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'name': 'img2mol-cddd-server',
    })


@app.route('/smiles_to_cddd/', methods=['POST'])
def smiles_to_cddd():
    data = json.loads(request.data)
    smiles = data["smiles"]
    smiles, _ = preprocess_smiles(smiles)
    embedding = INFERENCE_SERVER.seq_to_emb(smiles)
    embedding = embedding.tolist()
    return jsonify(embedding)


@app.route('/cddd_to_smiles/', methods=['POST'])
def cddd_to_smiles():
    data = json.loads(request.data)
    smiles = INFERENCE_SERVER.emb_to_seq(np.array(data["cddd"]))
    return jsonify(smiles)


@app.route('/sample_cddd_neighborhood/', methods=['POST'])
def sample_cddd_neighborhood():
    data = json.loads(request.data)
    smiles, _ = preprocess_smiles(data["smiles"])
    embedding = INFERENCE_SERVER.seq_to_emb(smiles)
    shape = embedding.shape
    shift = np.random.normal(
        scale=data["std"],
        size=(data["num_samples"], shape[-1]))
    embedding = np.tile(embedding, (data["num_samples"], 1))
    embedding += shift
    smiles = INFERENCE_SERVER.emb_to_seq(embedding)
    smiles = list(set(smiles))
    smiles, error = preprocess_smiles(smiles)
    smiles = [smiles[i] for i in range(len(smiles)) if error[i] == 0]
    smiles = list(set(smiles))
    return jsonify(smiles)


def preprocess_smiles(smiles):
    new_smiles = []
    error = []
    if isinstance(smiles, str):
        smiles = [smiles]
    for sml in smiles:
        mol = Chem.MolFromSmiles(sml)
        if mol is not None:
            new_sml = Chem.MolToSmiles(mol)
            new_sml2 = remove_salt_stereo(new_sml, REMOVER)
            new_sml2 = organic_filter(new_sml2)
            if isinstance(new_sml2, str):
                new_smiles.append(new_sml2)
                error.append(0)
            else:
                new_smiles.append(new_sml)
                error.append(1)
        else:
            error.append(2)
            new_smiles.append(sml)
    return new_smiles, error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./default_model")
    parser.add_argument("--device", default="0", type=str, nargs="+")
    parser.add_argument("--servers_per_device", default=1, type=int)
    parser.add_argument("--inference_port_frontend", default=5520, type=int)
    parser.add_argument("--inference_port_backend", default=5521, type=int)
    parser.add_argument("--port_app", default=8892, type=int)
    parser.add_argument("--max_len", default=150, type=int)
    parser.add_argument("--use_running", dest='use_running', action='store_true')
    parser.add_argument("--num_top", default=1, type=int)
    parser.set_defaults(use_running=False)

    flags, _ = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(flags.device)
    num_servers = len(flags.device) * flags.servers_per_device

    INFERENCE_SERVER = InferenceServer(
        use_running=flags.use_running,
        num_servers=num_servers,
        maximum_iterations=flags.max_len,
        port_frontend=flags.inference_port_frontend,
        port_backend=flags.inference_port_backend,
        model_dir=flags.model_dir,
        num_top=flags.num_top
    )
    app.run(
        threaded=False,
        processes=2,
        debug=False,
        host='0.0.0.0',
        port=flags.port_app
    )
