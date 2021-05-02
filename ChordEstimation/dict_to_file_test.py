
import csv


def train_dict_to_file(results, path):
	metrics = ['melody_accuracy', 'melody_accuracy_topk', 'chord_accuracy', 'chord_accuracy_topk']


	# save logged informations into log dict
	log = {}
	for key, value in results.items():
		if key == 'metrics':
			log.update({'train_' + mtr : value[i] for i, mtr in enumerate(metrics)})
		elif key == 'valid_metrics':
			log.update({'valid_' + mtr : value[i] for i, mtr in enumerate(metrics)})
		else:
			log[key] = value

	print(log)

	# results.update({mtr : value[i] for i, mtr in enumerate(results['metrics'])})

	with open(path, 'w') as f:
		w = csv.DictWriter(f, log.keys())
		w.writeheader()
		w.writerow(log)

def test_dict_to_file(results, path):
	with open(path, 'w') as f:
		w = csv.DictWriter(f, results.keys())
		w.writeheader()
		w.writerow(results)

train_results = {'loss': 2.5633039474487305, 'metrics': [0.11772916398272064, 0.3094135680422161, 1.0, 1.0], 'val_loss': 2.522510290145874, 'valid_metrics': [0.153919923279789, 0.31407336370175015, 1.0, 1.0]}
test_results = {'loss': 2.719392062565468, 'melody_accuracy': 0.0022572835064467106, 'melody_accuracy_topk': 0.005573055597656932, 'chord_accuracy': 0.01675977653631285, 'chord_accuracy_topk': 0.01675977653631285}

train_dict_results_filename = 'train_dict_results.csv'
test_dict_results_filename = 'test_dict_results.csv'

train_dict_to_file(train_results, train_dict_results_filename)
test_dict_to_file(test_results, test_dict_results_filename)

