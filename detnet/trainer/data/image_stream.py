from pathlib import Path
from PIL import Image
from torch.utils.data import IterableDataset


class TransformedStream(IterableDataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __iter__(self):
        for sample in self.dataset:
            yield self.transform(sample)


class ImageStream(IterableDataset):
    def __init__(self, path, interval=1):
        self.interval = interval
        self.videos = []
        self.images = []

        if isinstance(path, int):
            # try to open camera device
            self.videos = [path]
        elif isinstance(path, str):
            if path.isnumeric():
                # camera device
                self.videos = [int(path)]
            elif path.startswith('http'):
                import pafy
                video_pafy = pafy.new(path)
                print(video_pafy.title)
                best = video_pafy.getbest()
                self.videos.append(best.url)
            else:
                path = Path(path)
                if path.is_dir():
                    self.images = path.glob("**/*.jpg")
                    self.videos = [str(p) for p in path.glob("**/*.mp4")]
                elif path.suffix.lower() in ('.jpg',):
                    self.images = [path]
                elif path.suffix.lower() in ('.mp4', '.webm'):
                    self.videos = [str(path)]
        assert self.videos or self.images

    def __iter__(self):
        for i, image in enumerate(self.images):
            if i % self.interval == 0:
                yield {'image_id': image, 'input': Image.open(image), 'file_name': image}
        for video in self.videos:
            import cv2
            reader = cv2.VideoCapture(video)
            frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            i = 0
            while True:
                _, image = reader.read()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if i % self.interval == 0:
                    yield {'image_id': f"{video}_{i}", 'input': Image.fromarray(image), 'file_name': video}
                i += 1
                if 0 < frame_count <= i:
                    break

    def __rshift__(self, other):
        """transformed_dataset = dataset >> transform"""
        if not callable(other):
            raise RuntimeError('Dataset >> callable only!')
        return TransformedStream(dataset=self, transform=other)
