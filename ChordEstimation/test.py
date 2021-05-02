import os
import json
import torch
import argparse
from data_loader.data_loaders import TestMidiDataLoader
import model.model as module_model
from utils import get_instance
from tqdm import tqdm

# def _eval_metrics(metrics, output, target, extra=None):
#     acc_metrics = np.zeros(len(metrics))
#     for i, metric in enumerate(metrics):
#         acc_metrics[i] += metric(output, target, extra=extra)
#         # self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
#     return acc_metrics

def test_model(config, resume):
    # setup data_loader instances
    data_loader = TestMidiDataLoader(data_path='./data/small_key_output.pkl')

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
                
            print(f'data: {data}')
            print(f'target: {target}')
            print(f'extra: {extra}')
            output = model(data, extra)
            #
            # save sample song sequences or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target, extra)


            for k in output.keys():
                output[k] = output[k].cpu()
            for k in target.keys():
                target[k] = target[k].cpu()
            # total_metrics += _eval_metrics(output, target, extra=extra)



            batch_size = data_loader.batch_size
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target, extra) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
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
