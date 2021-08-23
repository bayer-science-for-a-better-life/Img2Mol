# Copyright 2021 Machine Learning Research @ Bayer AG
#
# Licensed for non-commercial use only, under the terms of the
# Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.
# You can find details at: https://creativecommons.org/licenses/by-nc/4.0/legalcode


import json
import requests
requests.packages.urllib3.disable_warnings()

"""
CDDD Server to encode SMILES string to molecular embeddings and decode the molecular embeddings to SMILES string.

For further details, please refer to: 
    [1] R. Winter, F. Montanari, F. Noe and D. Clevert, Chem. Sci, 2019,
     https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc04175j#!divAbstract

and: https://github.com/jrwnter/cddd
"""

# Note that the DEFAULT_HOST is accessing the AWS instance deployed by Machine Learning Research Group of Bayer.
DEFAULT_HOST = "http://ec2-18-157-240-87.eu-central-1.compute.amazonaws.com"

"""
The CDDD server is applicable for non-commercial use only, under the terms of the
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.
You can find details at: https://creativecommons.org/licenses/by-nc/4.0/legalcode
"""


class CDDDRequest:
    def __init__(self, host=DEFAULT_HOST, port=8892):
        self.host = host
        self.port = port
        self.headers = {'content-type': 'application/json'}

    def smiles_to_cddd(self, smiles):
        url = "{}:{}/smiles_to_cddd/".format(self.host, self.port)
        req = json.dumps({"smiles": smiles})
        response = requests.post(url, data=req, headers=self.headers, verify=False)
        return json.loads(response.content.decode("utf-8"))

    def seq_to_emb(self, smiles):
        return self.smiles_to_cddd(smiles)

    def cddd_to_smiles(self, embedding):
        url = "{}:{}/cddd_to_smiles/".format(self.host, self.port)
        req = json.dumps({"cddd": embedding})
        response = requests.post(url, data=req, headers=self.headers, verify=False)
        return json.loads(response.content.decode("utf-8"))

    def emb_to_seq(self, embedding):
        return self.cddd_to_smiles(embedding)