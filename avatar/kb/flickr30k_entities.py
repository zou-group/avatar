import os
import json
import os.path as osp
from PIL import Image
from avatar.utils.flickr30k_entities_utils import get_annotations, get_sentence_data
from avatar.utils.process_image import extract_patch


class Flickr30kEntities:
    """
    A class to handle the Flickr30k Entities dataset, including loading and processing images and annotations.

    Args:
        root (str): The root directory of the dataset.
    """

    def __init__(self, root: str):
        """
        Initializes the Flickr30kEntities class.

        Args:
            root (str): The root directory of the dataset.
        """
        self.root = osp.join(root, "flickr30k_entities")
        self.processed_dir = osp.join(self.root, "processed")
        self.raw_dir = osp.join(self.root, "raw")
        self.split_dir = osp.join(self.root, "split")
        if not osp.exists(self.processed_dir):
            assert osp.exists(
                self.raw_dir
            ), f"Please download the dataset from {self.raw_dir}"
            self.process()
        self.indices = [
            int(f.split("_")[-1].split(".")[0]) for f in os.listdir(self.processed_dir)
        ]
        self.indices.sort()
        self.candidate_ids = self.indices
        self.num_candidates = len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Gets the data for the image at the specified index.

        Args:
            idx (int): The index of the image.

        Returns:
            dict: The data for the image.
        """
        image_id = self.indices[idx]
        with open(osp.join(self.processed_dir, f"image_{image_id}.json"), "r") as f:
            data = json.load(f)
        return data

    def __len__(self) -> int:
        """
        Gets the number of images in the processed directory.

        Returns:
            int: The number of images.
        """
        return len(os.listdir(self.processed_dir))

    def get_data_by_id(self, image_id: int) -> dict:
        """
        Gets the data for the image with the specified ID.

        Args:
            image_id (int): The ID of the image.

        Returns:
            dict: The data for the image.
        """
        with open(osp.join(self.processed_dir, f"image_{image_id}.json"), "r") as f:
            data = json.load(f)
        return data

    def get_doc_info(self, image_id: int, **kwargs) -> str:
        """
        Gets the complete textual and relational information for the image with the specified ID.

        Args:
            image_id (int): The ID of the image.

        Returns:
            str: The complete textual and relational information for the image.
        """
        data = self.get_data_by_id(image_id)
        patches = data["patches"]

        # Bag of phrases
        bow = []
        for p in patches.values():
            bow.append("/".join(p["phrase"]))
        return "An image with entities: " + ", ".join(bow)

    def get_image(self, image_id: int) -> Image.Image:
        """
        Gets the image with the specified ID.

        Args:
            image_id (int): The ID of the image.

        Returns:
            Image.Image: The image.
        """
        relative_image_path = self.get_data_by_id(image_id)["relative_image_path"]
        image = Image.open(osp.join(self.root, relative_image_path))
        return image

    def get_patch(self, image_id: int, patch_id: int) -> Image.Image:
        """
        Gets a patch of the image with the specified ID and patch ID.

        Args:
            image_id (int): The ID of the image.
            patch_id (int): The ID of the patch.

        Returns:
            Image.Image: The patch of the image.
        """
        image = self.get_image(image_id)
        box = self.get_data_by_id(image_id)["patches"][str(patch_id)]["position"]
        patch = extract_patch(image, box[0])
        return patch

    def patch_id_to_phrase_dict(self, image_id: int) -> dict:
        """
        Gets a dictionary mapping patch IDs to phrases for the image with the specified ID.

        Args:
            image_id (int): The ID of the image.

        Returns:
            dict: A dictionary mapping patch IDs to phrases.
        """
        data = self.get_data_by_id(image_id)
        patch_to_phrase = {}
        for patch_id, patch_info in data["patches"].items():
            if len(patch_info["position"]):
                patch_to_phrase[int(patch_id)] = patch_info["phrase"]
        return patch_to_phrase

    def process(self):
        """
        Processes the raw dataset and creates the processed dataset.
        """

        def process_one(
            image_id: int, exclude_sentence_idx: int = 0, collect_all: bool = True
        ) -> dict:
            """
            Processes a single image and its annotations.

            Args:
                image_id (int): The ID of the image.
                exclude_sentence_idx (int, optional): The index of the sentence to exclude. Default is 0.
                collect_all (bool, optional): Whether to collect all sentences. Default is True.

            Returns:
                dict: The processed data for the image.
            """
            phrases = {}
            sentence = get_sentence_data(
                osp.join(self.raw_dir, f"Sentences/{image_id}.txt")
            )
            annotation = get_annotations(
                osp.join(self.raw_dir, f"Annotations/{image_id}.xml")
            )
            for i, s in enumerate(sentence):
                if i == exclude_sentence_idx:
                    continue
                for phrase in s["phrases"]:
                    if int(phrase["phrase_id"]) in phrases:
                        phrases[int(phrase["phrase_id"])] = {
                            "phrase": phrases[int(phrase["phrase_id"])]["phrase"]
                            + [phrase["phrase"].lower()],
                            "type": phrase["phrase_type"],
                        }
                    else:
                        phrases[int(phrase["phrase_id"])] = {
                            "phrase": [phrase["phrase"].lower()],
                            "type": phrase["phrase_type"],
                        }
                if not collect_all:
                    break
            for phrase_id, phrase in phrases.items():
                phrases[phrase_id]["phrase"] = list(set(phrases[phrase_id]["phrase"]))
                phrases[phrase_id]["box"] = []
            for phrase_id, box in annotation["boxes"].items():
                if int(phrase_id) in phrases:
                    phrases[int(phrase_id)]["box"] = box
            phrases["idx"] = image_id
            phrases["relative_image_path"] = osp.join(
                f"raw/flickr30k-images/{image_id}.jpg"
            )
            phrases["image_size"] = {
                "width": annotation["width"],
                "height": annotation["height"],
                "depth": annotation["depth"],
            }
            return phrases

        for split in ["train", "val", "test"]:
            with open(os.path.join(self.split_dir, f"{split}.index"), "r") as f:
                for idx, line in enumerate(f):
                    image_id = int(line.strip())
                    data = process_one(image_id=image_id, collect_all=False)
                    os.makedirs(self.processed_dir, exist_ok=True)
                    with open(
                        osp.join(self.processed_dir, f"image_{image_id}.json"), "w"
                    ) as f:
                        json.dump(data, f, indent=4)
