# covid_data

The activities of solidary economy in Catalunya during the COVID19 crisis.


In order to run the analysis notebook you will need to follow the next steps:

1. Install all the necesary libraries:
```bash {cmd}
pip install -r requirements.txt
```
2. Enable the notebook extensions so the Sankey diagrams are shown:
```bash {cmd}
jupyter nbextension enable --py --sys-prefix ipysankeywidget
```
3. Start the notebook server from the command line:
```bash {cmd}
jupyter notebook
```
4. Open `analysis.ipynb` on the web browser and run it. The second cell in the notebook will ask if you want to compute the pre-process of the input files. If it's the first time running the code you will need to input 'y' in the window that will appear, input 'n' otherwise.


