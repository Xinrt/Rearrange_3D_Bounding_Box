from Dataset import MyDataset, custom_collate_fn
from torch.utils.data import DataLoader

DATASET_PATH = '/vast/xt2191/dataset'

if __name__ == "__main__":
    # may take a long time since we have too many files
    dataset = MyDataset(DATASET_PATH)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    print("Length: ", len(dataset))

    for batch in dataloader:
        print(batch['rgb'])
        print("annotation: ", batch['annotation'])

        print("rgb.shape: ", batch['rgb'].shape)
        print("depth.shape: ", batch['depth'].shape)
        print("segmentation.shape: ", batch['sem'].shape)
        # print("pan.shape: ", batch['pan'].shape)
        break
        