import os
import sys
import csv
import json
from train import train_model
from test import test_model

def train_dict_to_file(results, path):
	metrics = ['melody_accuracy', 'melody_accuracy_topk', 'chord_accuracy', 'chord_accuracy_topk']

	flattened = {}
	for key, value in results.items():
		if key == 'metrics':
			flattened.update({'train_' + mtr : value[i] for i, mtr in enumerate(metrics)})
		elif key == 'valid_metrics':
			flattened.update({'valid_' + mtr : value[i] for i, mtr in enumerate(metrics)})
		else:
			flattened[key] = value

	print('Saving train results: ', flattened)
	dict_to_file(flattened, path)

def dict_to_file(results, path):
	with open(path, 'w') as f:
		w = csv.DictWriter(f, results.keys())
		w.writeheader()
		w.writerow(results)

def just_train(config, train_data_path):
	config_json = json.load(open(config))
	train_data = train_model(config_json, None, train_data_path)

	test_run_path = train_data['checkpoint_dir']
	saved_training_results_path = train_data['checkpoint_dir'] + 'train_results.csv'
	train_results = train_data['log']


	print('Saving train results')
	train_dict_to_file(train_results, saved_training_results_path)

	return test_run_path



def just_test(test_run_path, test_data_path):

	config_json = json.load(open(os.path.join(test_run_path, 'config.json')))
	print(config_json)
	test_results = test_model(config_json, os.path.join(test_run_path, 'model_best.pth'), test_data_path)

	saved_testing_results_path = test_run_path + 'test_results.csv'
	print('Saving test results')
	print(test_results)
	dict_to_file(test_results, saved_testing_results_path)


def train_and_test(config, train_data_path='data/small_train_dataset.pkl', test_data_path='data/small_test_dataset.pkl'):
	print('Starting training on ', config)
	
	test_run_path = just_train(config, train_data_path)
	
	print('\nStarting testing on ', test_run_path + 'config.json')
	just_test(test_run_path, test_data_path)



if __name__ == '__main__':
	configs = os.listdir('configs')
	configs.reverse()
	print(configs)

	# train_dataset = 'data/large_train_dataset.pkl'
	# test_dataset = 'data/large_test_dataset.pkl'
	# train_dataset = 'data/small_train_dataset.pkl'
	# test_dataset = 'data/small_test_dataset.pkl'

	train_dataset = 'data/train_out.pkl'
	test_dataset = 'data/test_out.pkl'
	for i, config in enumerate(configs):
		print(f'\n\nStarting Model {i} out of {len(configs)}')
		try:
			import torch
			torch.cuda.empty_cache()
			train_and_test(os.path.join('./configs/', config), train_dataset, test_dataset)
		except:
			print("Unexpected error:", sys.exc_info())


	# test_run_dir = "saved/64_Batch_10_worker_Combined_BiLSTM_01`\`0502_051351`\`"
	# test_config = open(os.path.join(test_run_dir, 'config.json') 
	# test_run_path = test_run_dir
	# just_test(test_run_path)
