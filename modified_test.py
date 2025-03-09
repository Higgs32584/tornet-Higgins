from tornet.data.loader import get_dataloader
import sys

def main():
    dataloader=config.get('dataloader')
    ds_train = get_dataloader(dataloader, DATA_ROOT, train_years, "train", batch_size, weights, **dataloader_kwargs)





if __name__=='__main__':
    config=''
            # Load param file if given
    if len(sys.argv)>1:
        config.update(json.load(open(sys.argv[1],'r')))
        main(config)
