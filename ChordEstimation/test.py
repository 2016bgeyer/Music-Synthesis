import os
import json
import torch
import argparse
from data_loader.data_loaders import TestMidiDataLoader
import model.model as module_model
from utils import get_instance
from tqdm import tqdm

def test_model(config, resume, data_path='./data/small_test_dataset.pkl'):
    # setup data_loader instances
    data_loader = TestMidiDataLoader(data_path)

    # build model architecture
    model = get_instance(module_model, 'model', config)
    model.summary()

    # get function handles of loss and metrics
    loss_fn = getattr(module_model, config['loss'])
    metric_fns = [getattr(module_model, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target, extra) in enumerate(tqdm(data_loader)):
            for k in data.keys():
                data[k] = data[k].to(device)
            for k in target.keys():
                target[k] = target[k].to(device)

            output = model(data, extra)
            #
            # save sample song sequences or do something with output here
            #

            loss = loss_fn(output, target, extra)

            for k in output.keys():
                output[k] = output[k].cpu()
            for k in target.keys():
                target[k] = target[k].cpu()

            total_loss += loss.item() * data_loader.batch_size # multiply all by batch size instead of each one independently
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target, extra)

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__ : total_metrics[i].item() for i, met in enumerate(metric_fns)})
    print(log)
    return log

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='config/config.json', type=str,
                           help='config file path (default: config/config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        # path = os.path.join(config['trainer']['save_dir'], config['name'])
    else:
        raise ValueError("Configuration file need to be specified. Add '-c config.json', for example.")

    test_model(config, args.resume)
