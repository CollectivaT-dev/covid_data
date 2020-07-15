import preprocess as prep
import location_utils as loc


input_f = prep.check_input_files()

if input_f:
	print('Reading the initial files')
	pagesos, abastiment, data_gen, locations, stopwords, paths, conf = prep.read_initial_data()

	print('Getting "comarca" column ready for each file')
	pagesos, abastiment, data_gen = loc.all_comarca_column_prep(pagesos, abastiment, data_gen, 
		locations, stopwords, conf['comarca_typos'])

	print('Pre-processing columns')
	pagesos, abastiment, data_gen = prep.all_prep_dataset(pagesos, abastiment, data_gen, locations,
		stopwords)

	print('Add new columns to the datasets')
	pagesos, abastiment, data_gen = prep.all_add_new_cols(pagesos, abastiment, data_gen, locations, 
	    conf)
	pagesos, abastiment, data_gen = prep.all_add_num_cols(pagesos, abastiment, data_gen, conf)

	#save_data_set(pagesos, abastiment, data_gen, paths['input_path'])

	print('Find the duplicates in the datasets and merging the data')
	covid_data = prep.merge_covid_data(pagesos, abastiment, locations, conf)
	covid_data = prep.covid_in_gen(covid_data, data_gen, conf)

	print('Obtaining the coordinates of all the comarques')
	covid_data, com_coord = loc.get_comarca_coords(covid_data)

	print('Saving the data')
	prep.save_merged_data(covid_data, data_gen, com_coord, paths)


	#covid_data, data_gen, com_coord, cat = prep.read_final_data()

