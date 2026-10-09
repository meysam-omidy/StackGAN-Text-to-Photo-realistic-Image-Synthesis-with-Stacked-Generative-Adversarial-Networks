"""Quick integrity and content check for the CUB/StackGAN dataset."""

from utils import BirdTextDataset


def main() -> None:
    for split in ("train", "test"):
        dataset = BirdTextDataset(split, image_size=64, training=False)
        image, embedding, _ = dataset[0]
        print(f"{split}: {len(dataset)} image/text pairs; image={tuple(image.shape)}; embedding={tuple(embedding.shape)}")
        print(f"  image: {dataset.image_paths[0]}")
        for caption in dataset.caption(0):
            print(f"  caption: {caption}")


if __name__ == "__main__":
    main()
