import os


def download_common_models_for_pytorch_transformers(cache_dir=None):
    if cache_dir is None:
        cache_dir = os.environ['TRANSFORMERS_CACHE']
    from pytorch_transformers import BertTokenizer
    BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
    BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=cache_dir)
