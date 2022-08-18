## example-deployment

Some example code on how to run the deployed img2mol app.


```shell
# run the cddd server
(cd cddd_server/src; ./download_default_model.sh)
conda env create --file cddd_server/env.cddd-server.explicit.yaml
conda run -n cddd-server python -m pip install -r cddd_server/env.cddd-server.requirements.txt
conda run -n cddd-server --cwd ./cddd-server/src python server.py --model_dir ./default_model

# run the img2mol server
(cd img2mol_server/src; ./download_model.sh)
conda env create --file cddd_server/env.img2mol-server.explicit.yaml
conda run -n img2mol-server python -m pip install -r cddd_server/env.img2mol-server.requirements.txt
conda run -n img2mol-server --cwd ./img2mol-server/src python main.py

# run the img2mol frontend
conda env create --file img2mol_frontend/env.img2mol_frontend.explicit.yaml
conda run -n img2mol-frontend python -m pip install -r cddd_server/env.img2mol_frontend.requirements.txt
conda run -n img2mol-frontend --cwd ./img2mol_frontend/src streamlit run app.py
```
