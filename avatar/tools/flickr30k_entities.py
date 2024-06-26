from typing import List, Dict
from avatar.tools.tool import Tool
from avatar.utils.format import format_checked
import PIL


class GetBagOfPhrases(Tool):
    """
    A class to retrieve a bag of phrases for a list of image IDs.

    Args:
        kb: The knowledge base containing the image information.
    """

    def __init__(self, kb, **kwargs):
        super().__init__(kb=kb)
        assert hasattr(kb, 'patch_id_to_phrase_dict'), "kb must have a method 'patch_id_to_phrase_dict'"

    @format_checked
    def __call__(self, image_ids: List[int]) -> List[List[str]]:
        """
        Retrieves a bag of phrases for a list of image IDs.

        Args:
            image_ids (List[int]): The list of image IDs.

        Returns:
            List[List[str]]: A list of lists of phrases for each image.
        """
        bag_of_phrases = []
        for image_id in image_ids:
            phrase_dict = self.kb.patch_id_to_phrase_dict(image_id)
            phrases = [phrase for phrase_lst in phrase_dict.values() for phrase in phrase_lst]
            bag_of_phrases.append(phrases)

        return bag_of_phrases
            
    def __str__(self):
        return 'get_bag_of_phrases(image_ids: List[int]) -> bag_of_phrases: List[List[str]]'
    
    def __repr__(self):
        return 'Returns a list of phrase list for each image in the image_ids list. For example, get_bag_of_phrases([20, 30]) -> [["a dog", "a puppy", "a cat"], ["a beautiful hat", "a white dress", "wedding dress"]]. Note that an entity may be repeated in the list with different phrases, such as "a dog" and "a puppy".'


class GetFlickrTextInfo(Tool):
    """
    A class to retrieve the text information for a list of image IDs.

    Args:
        kb: The knowledge base containing the image information.
    """

    def __init__(self, kb, **kwargs):
        super().__init__(kb=kb)
        assert hasattr(kb, 'get_doc_info'), "kb must have a method 'get_doc_info'"

    @format_checked
    def __call__(self, image_ids: List[int]) -> List[str]:
        """
        Retrieves the text information for a list of image IDs.

        Args:
            image_ids (List[int]): The list of image IDs.

        Returns:
            List[str]: A list of text information for each image.
        """
        texts = []
        for image_id in image_ids:
            text = self.kb.get_doc_info(image_id)
            texts.append(text)
        
        return texts

    def __str__(self):
        return 'get_text_info(image_ids: List[int]) -> texts: List[str]'

    def __repr__(self):
        return f'Returns a list of text information for each image in the image_ids list. For example, get_text_info([20, 30]) -> ["An image with entities: a dog/a puppy, a cat", "An image with entities: a beautiful hat, a white dress/wedding dress"]'


class GetImages(Tool):
    """
    A class to retrieve the image objects for a list of image IDs.

    Args:
        kb: The knowledge base containing the image information.
    """

    def __init__(self, kb, **kwargs):
        super().__init__(kb=kb)
        assert hasattr(kb, 'get_image'), "kb must have a method 'get_image'"

    @format_checked
    def __call__(self, image_ids: List[int]) -> List[PIL.Image.Image]:
        """
        Retrieves the image objects for a list of image IDs.

        Args:
            image_ids (List[int]): The list of image IDs.

        Returns:
            List[PIL.Image.Image]: A list of image objects for each image ID.
        """
        images = []
        for image_id in image_ids:
            image = self.kb.get_image(image_id)
            images.append(image)

        return images
            
    def __str__(self):
        return 'get_images(image_ids: List[int]) -> images: List[PIL.Image.Image]'
    
    def __repr__(self):
        return f'Returns a list of Image objects for each image in the image_ids list. For example, get_images([20, 30]) -> [Image, Image]'


class GetPatchID2PhraseDict(Tool):
    """
    A class to retrieve the patch ID to phrase dictionary for a list of image IDs.

    Args:
        kb: The knowledge base containing the image information.
    """

    def __init__(self, kb, **kwargs):
        super().__init__(kb=kb)
        assert hasattr(kb, 'patch_id_to_phrase_dict'), "kb must have a method 'patch_id_to_phrase_dict'"

    @format_checked
    def __call__(self, image_ids: List[int]) -> List[Dict[int, List[str]]]:
        """
        Retrieves the patch ID to phrase dictionary for a list of image IDs.

        Args:
            image_ids (List[int]): The list of image IDs.

        Returns:
            List[Dict[int, List[str]]]: A list of dictionaries mapping patch IDs to phrases for each image.
        """
        list_of_patch_id_to_phrase_dict = []
        for image_id in image_ids:
            patch_to_phrase_dict = self.kb.patch_id_to_phrase_dict(image_id)
            list_of_patch_id_to_phrase_dict.append(patch_to_phrase_dict)

        return list_of_patch_id_to_phrase_dict
            
    def __str__(self):
        return 'get_patch_id_to_phrase_dict(image_ids: List[int]) -> list_of_patch_id_to_phrase_dict: List[Dict[int, List[str]]]'
    
    def __repr__(self):
        return 'Returns a list of patch_id to phrase list dictionary for each image in the image_ids list. For example, get_patch_id_to_phrase_dict([20, 30]) -> [{201: ["a dog", "a puppy"], 202: ["a cat"]} , {301: ["a beautiful hat"], 302: ["a white dress", "wedding dress"]}]. Note that the patches may have the same entity with different phrases, such as "a dog" and "a puppy", and each dictionary may only contain the patches of a subset of entities in the image.'


class GetImagePatchByPhraseId(Tool):
    """
    A class to retrieve the patch image for a given image ID and patch ID.

    Args:
        kb: The knowledge base containing the image information.
    """

    def __init__(self, kb, **kwargs):
        super().__init__(kb=kb)
        assert hasattr(kb, 'get_patch'), "kb must have a method 'get_patch'"
    
    @format_checked
    def __call__(self, image_id: int, patch_id: int) -> PIL.Image.Image:
        """
        Retrieves the patch image for a given image ID and patch ID.

        Args:
            image_id (int): The ID of the image.
            patch_id (int): The ID of the patch.

        Returns:
            PIL.Image.Image: The patch image.
        """
        patch = self.kb.get_patch(image_id, patch_id)
        return patch
            
    def __str__(self):
        return 'get_image_patch_by_phrase_id(image_id: int, patch_id: int) -> patch: PIL.Image.Image'
    
    def __repr__(self):
        return f'Returns the patch image for the given image_id and patch_id. For example, get_image_patch_by_phrase_id(20, 201) -> Image'
