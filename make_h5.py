import h5py
import tqdm
import pathlib
from config import Config


def get_h5(data_file, h5_file, config):
    with h5py.File((config.H5_FOLDER / h5_file), "w") as h5f:
        with open(data_file, "r") as file:
            num_lines = sum(1 for l in file)
        with open(data_file, "r") as file:
            pbar = tqdm.tqdm(total=num_lines)
            for i, row in enumerate(file):
                h5f.create_group(str(i))
                h5f[str(i)].create_dataset('row', data=row)
                pbar.update(1)
            pbar.close()


if __name__ == "__main__":
    config = Config.get_default_config(None)

    config.H5_FOLDER.mkdir(exist_ok=True)

    print("Retrieving training h5....")
    get_h5(config.TRAIN_PATH, "train.h5", config)

    print("Retrieving validation h5....")
    get_h5(config.VAL_PATH, "val.h5", config)

    print("Retrieving testing h5....")
    get_h5(config.TEST_PATH, "test.h5", config)