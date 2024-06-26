import PIL
import torch
from typing import List, Union
from transformers import AutoTokenizer, AutoProcessor, CLIPModel, CLIPTextConfig

from avatar.utils.format import format_checked
from avatar.tools.tool import Tool


class GetCLIPTextEmbedding(Tool):
    """
    Class to get CLIP text embeddings.

    Args:
        emb_model (str): The pre-trained CLIP model to use. Default is "openai/clip-vit-large-patch14".
        batch_size (int): The batch size for processing. Default is 4.
        use_cuda (bool): Whether to use CUDA for processing. Default is True
        **kwargs: Additional arguments.
    """

    def __init__(self, 
                 emb_model: str = "openai/clip-vit-large-patch14", 
                 batch_size: int = 4, 
                 use_cuda: bool = True,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.model = CLIPModel.from_pretrained(emb_model)
        self.tokenizer = AutoTokenizer.from_pretrained(emb_model)
        text_config = CLIPTextConfig.from_pretrained(emb_model)
        self.max_length = text_config.max_position_embeddings
        if use_cuda:
            self.model = self.model.cuda()

    @format_checked
    def __call__(self, string: Union[str, List[str]]) -> torch.Tensor:
        """
        Generates CLIP text embeddings for the given string or list of strings.

        Args:
            string (Union[str, List[str]]): The input string or list of strings to embed.

        Returns:
            torch.Tensor: The generated embeddings.
        """
        if isinstance(string, str):
            string = [string]
        assert all(len(s) > 0 for s in string), 'Every string in the list to be embedded should be non-empty'

        print(f'get_clip_text_embedding - input {string}')
        text_embeds = []
        for text_batch in [string[i:i + self.batch_size] for i in range(0, len(string), self.batch_size)]:
            with torch.no_grad():
                inputs = self.tokenizer(text_batch, padding="max_length", truncation=True, 
                                        max_length=self.max_length, return_tensors="pt")
                inputs = {k: v.cuda() if self.use_cuda else v for k, v in inputs.items()}
                text_batch_embs = self.model.get_text_features(**inputs).cpu()
            text_embeds.append(text_batch_embs.view(len(text_batch), -1))
        text_embeds = torch.cat(text_embeds, dim=0)

        print(f'get_clip_text_embedding - output shape {text_embeds.size()}')
        return text_embeds

    def __str__(self):
        return 'get_clip_text_embedding(string: Union[str, List[str]]) -> embedding: torch.Tensor'

    def __repr__(self):
        return ("Embed a string or list of N strings into a tensor of size (N, hidden_dim). For efficiency, "
                "include multiple strings in the list at once, rather than calling the function separately "
                "for each string.")


class GetCLIPImageEmbedding(Tool):
    """
    Class to get CLIP image embeddings.

    Args:
        emb_model (str): The pre-trained CLIP model to use. Default is "openai/clip-vit-large-patch14".
        batch_size (int): The batch size for processing. Default is 4.
        use_cuda (bool): Whether to use CUDA for processing. Default is True
        **kwargs: Additional arguments.
    """

    def __init__(self, 
                 emb_model: str = "openai/clip-vit-large-patch14", 
                 batch_size: int = 4, 
                 use_cuda: bool = True,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.model = CLIPModel.from_pretrained(emb_model)
        self.processor = AutoProcessor.from_pretrained(emb_model)
        if use_cuda:
            self.model = self.model.cuda()

    @format_checked
    def __call__(self, image_lst: List[PIL.Image.Image]) -> torch.Tensor:
        """
        Generates CLIP image embeddings for the given list of images.

        Args:
            image_lst (List[PIL.Image.Image]): The list of images to embed.

        Returns:
            torch.Tensor: The generated embeddings.
        """
        print(f'get_clip_image_embedding - len(image_lst) {len(image_lst)}')
        image_embs = []
        for image_batch in [image_lst[i:i + self.batch_size] for i in range(0, len(image_lst), self.batch_size)]:
            with torch.no_grad():
                inputs = self.processor(images=image_batch, return_tensors="pt")
                inputs = {k: v.cuda() if self.use_cuda else v for k, v in inputs.items()}
                image_batch_embs = self.model.get_image_features(**inputs).cpu()
            image_embs.append(image_batch_embs.view(len(image_batch), -1))
        image_embs = torch.cat(image_embs, dim=0)

        print(f'get_clip_image_embedding - output shape {image_embs.size()}')
        return image_embs

    def __str__(self):
        return 'get_clip_image_embedding(image_lst: List[PIL.Image.Image]) -> embedding: torch.Tensor'

    def __repr__(self):
        return ("Embed a list of images into a tensor of size (len(image_lst), hidden_dim). "
                "For example, get_image_embedding([image1, image2]) returns a tensor of size (2, hidden_dim).")
