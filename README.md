# covid_data

The activities of solidary economy in Catalunya during the COVID19 crisis.

To execute the Jupyter Notebooks you will need some extra files:
- abastiment.csv (data from the campaign "Abastiment Agroecològic" of Arran de Terra)
- db_mesinfo.json (data from the campaign "Pagesia a Casa" of Unió de Pagesos)
- Productors_adherits_a_la_venda_de_proximitat.csv (dataset of producers certified by Generalitat with "venta de proximidad")
- municipis_merge.csv
- catalunya3.png (mapa con las comarcas de Catalunya)
[- w2v_cat_model.bin (a pretrained Catalan word2vec)]
- paths.yaml (containing the "input_path" key with the local path where the data is located)



Order of execution of the notebooks:
1. covid_processat.ipynb: it cleans and process abastiment and generalitat data
2. pagesos_processat.ipynb: it cleans and process pagesos data
3. merge_datasets.ipynb: it merges abastiment and pagesos data in a single dataset
4. analysis.ipynb: it analises the response-to-covid-data (abastiment+pagesos) and compare them to pre-covid-data (generalitat)

