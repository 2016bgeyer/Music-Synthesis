import os
import csv
import json
from train import train_model
from test import test_model

def dict_to_file(results, path):
	with open(path, 'w') as f:
		w = csv.DictWriter(f, results.keys())
		w.writeheader()
		w.writerow(results)

def just_train(config):
	config_json = json.load(open(config))
	train_data = train_model(config_json, None)

	test_run_path = train_data['checkpoint_dir']
	saved_training_results_path = train_data['checkpoint_dir'] + 'train_results.csv'
	train_results = train_data['log']


	print('Saving train results')
	print(train_results)
	dict_to_file(train_results, saved_training_results_path)

	return test_run_path



def just_test(test_run_path):

	config_json = json.load(open(os.path.join(test_run_path, 'config.json')))
	print(config_json)
	test_results = test_model(config_json, os.path.join(test_run_path, 'model_best.pth'))

	saved_testing_results_path = test_run_path + 'test_results.csv'
	print('Saving test results')
	print(test_results)
	dict_to_file(test_results, saved_testing_results_path)


def train_and_test(config):
	print('Starting training on ', config)
	
	test_run_path = just_train(config)
	
	print('\nStarting testing on ', test_run_path + 'config.json')
	just_test(test_run_path)

if __name__ == '__main__':
	configs = ['my_config.json']

	for i, config in enumerate(configs):
		print(f'\n\nStarting Model {i} out of {len(configs)}')
		train_and_test(config)


	# test_run_dir = "saved/64_Batch_10_worker_Combined_BiLSTM_01`\`0502_051351`\`"
	# test_config = open(os.path.join(test_run_dir, 'config.json') 
	# test_run_path = test_run_dir
	# just_test(test_run_path)
