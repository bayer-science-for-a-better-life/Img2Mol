
import json

import requests

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


class CDDDRequest:
    def __init__(self, host, port, protocol="http"):
        self.host = host
        self.port = port
        self.protocol = protocol
        self.headers = {'content-type': 'application/json'}

    def smiles_to_cddd(self, smiles):
        url = "{}://{}:{}/smiles_to_cddd/".format(self.protocol, self.host, self.port)
        req = json.dumps({"smiles": smiles})
        response = requests.post(url, data=req, headers=self.headers, verify=False)
        return json.loads(response.content.decode("utf-8"))

    def cddd_to_smiles(self, embedding):
        url = "{}://{}:{}/cddd_to_smiles/".format(self.protocol, self.host, self.port)
        req = json.dumps({"cddd": embedding})
        response = requests.post(url, data=req, headers=self.headers, verify=False)
        return json.loads(response.content.decode("utf-8"))


def returnCanonicalSmiles(smi):
    mol = Chem.MolFromSmiles(smi, sanitize=True)
    if mol:
        return Chem.MolToSmiles(mol)
    else:
        return 'No valid SMILES recognized'
