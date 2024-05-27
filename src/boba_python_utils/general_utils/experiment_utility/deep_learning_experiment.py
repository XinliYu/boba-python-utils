from boba_python_utils.general_utils.experiment_utility.experiment_base import ExpArgInfo, ExperimentBase
from os import path

# region deep learning common arguments

_ARG_DEFAULT_GPU_INDEX = '*'
_ARG_DEFAULT_BATCH_SIZE = 64
_ARG_DEFAULT_BATCH_CACHE_GROUP_SIZE = 64
_ARG_DEFAULT_TEST_BATCH_SIZE = -1
_ARG_DEFAULT_EPOCHS = 20
_ARG_DEFAULT_LEARNING_RATE = 0.0004
_ARG_DEFAULT_REG_LAMBDA = 1E-4
_ARG_DEFAULT_MODEL_TEST_TOP_K = 3
_ARG_DEFAULT_WEIGHT_DECAY = 5E-4
_ARG_DEFAULT_DROPOUT = 0
_ARG_DEFAULT_EARLY_STOP_PATIENCE = 10
_ARG_DEFAULT_INPUT_STATE_DIM = 300
_ARG_DEFAULT_HIDDEN_STATE_DIM = [512, 256, 128]
_ARG_DEFAULT_MAX_SEQ_LEN = 12
_ARG_DEFAULT_SIMILARITY_LAYER = 'dot'
_ARG_DEFAULT_POOL_LAYER = 'sum'
_ARG_DEFAULT_EMBEDDING_SAVE_OPTION = 'numpy'
_ARG_DEFAULT_OUTPUT_SAVE_RATIO = 0.0
_ARG_DEFAULT_MODEL_SAVE_RATIO = 0.5
_ARG_DEFAULT_MODEL_SAVE_MIN = 3
ARG_SETUP_DEEP_LEARNING_GENERAL = (
    ExpArgInfo(full_name='gpu', default_value=_ARG_DEFAULT_GPU_INDEX, description='The index of the GPU to use.'),
    ExpArgInfo(full_name='batchsize', default_value=_ARG_DEFAULT_BATCH_SIZE, description='The training batch size.', affect_batch=True),
    ExpArgInfo(full_name='test_batchsize', default_value=_ARG_DEFAULT_BATCH_SIZE, description='The test batch size.', affect_batch=True),  # TODO at test time, the batchsize for the batch cache should be according to this parameter
    ExpArgInfo(full_name='epochs', default_value=_ARG_DEFAULT_EPOCHS, description='The number of training epochs'),
    ExpArgInfo(full_name='learning_rate', default_value=_ARG_DEFAULT_LEARNING_RATE, description='The learning rate. Rule of thumb: higher rate for simpler models; lower rate for complex models.'),
    ExpArgInfo(full_name='reg_lambda', default_value=_ARG_DEFAULT_REG_LAMBDA, description='The regularization penalty.'),
    ExpArgInfo(full_name='weight_decay', default_value=_ARG_DEFAULT_WEIGHT_DECAY, description='The optimization weight decay.'),
    ExpArgInfo(full_name='dropout', default_value=_ARG_DEFAULT_DROPOUT, description='The optimization weight decay.'),
    ExpArgInfo(full_name='early_stop_patience', default_value=_ARG_DEFAULT_EARLY_STOP_PATIENCE, description='During training, if the average performance declines for this number of previous epochs, '
                                                                                                            'then the early-stop is activated to terminate the entire training process without finishing the remaining epochs; '
                                                                                                            'to disable early-stop, assign 0 to this parameter'),
    ExpArgInfo(full_name='model_test_top_k', default_value=_ARG_DEFAULT_MODEL_TEST_TOP_K, description='Test the top-k models on the test sets.'),
    ExpArgInfo(full_name='input_state_dim', default_value=_ARG_DEFAULT_INPUT_STATE_DIM, description='The dimension of the input states (embedding).'),
    ExpArgInfo(full_name='hidden_state_dim', default_value=_ARG_DEFAULT_HIDDEN_STATE_DIM, description='The dimension of the hidden states (embeddings). May specify a list of integers for multiple layers of hidden state dimensions.'),
    ExpArgInfo(full_name='max_seq_len', default_value=_ARG_DEFAULT_MAX_SEQ_LEN, description='The maximum sequence length needed for sequence models.', affect_batch=True, affect_vocab=True),
    ExpArgInfo(full_name='similarity_layer', default_value=_ARG_DEFAULT_SIMILARITY_LAYER, description='The similarity function used in the neural network when applicable. A similarity layer is common in information retrieval models.'),
    ExpArgInfo(full_name='pool_layer', default_value=_ARG_DEFAULT_POOL_LAYER, description='The pooling layer used in the neural network when applicable. A pooling layer is common in NLP and computer vision models.'),
    ExpArgInfo(full_name='batch_cache_group_size', default_value=_ARG_DEFAULT_BATCH_CACHE_GROUP_SIZE, description='The number of batches in one cache group; (e.g. for a file cache, a cache group is typically written in a cache file).', affect_batch=True),
    ExpArgInfo(full_name='embedding_save_option', default_value=_ARG_DEFAULT_EMBEDDING_SAVE_OPTION, description='The options for embedding save. Currently support: 1) original, the embedding is saved as it is; 2) cpu, the embedding is saved as CPU tensors; 3) numpy, the embedding is saved as numpy arrays.'),
    ExpArgInfo(full_name='output_save_ratio', default_value=_ARG_DEFAULT_OUTPUT_SAVE_RATIO, description='Specifies the ratio of outputs to save when applying the model on the input data. The primary purpose of this parameter is to extract a percentage of model output for visualization or analysis.'),
    ExpArgInfo(full_name='model_save_ratio', default_value=_ARG_DEFAULT_OUTPUT_SAVE_RATIO, description='Specifies the ratio of models to save during the training. Typically, this ratio is with respect to the number of epochs. For example, when training for 10 epochs, and this ratio is 0.5, then we save models for the last 5 epochs.'),
    ExpArgInfo(full_name='model_save_min', default_value=_ARG_DEFAULT_MODEL_SAVE_MIN, description='Specifies the minimum number of models to save during the training. Use this parameter together with `model_save_ratio` to determine the actual number of saved models.')
)
EXP_BATCH_ARGS_DEEP_LEARNING_GENERAL = ('max_seq_len', 'batchsize', 'batch_cache_group_size')
EXP_NAME_ARGS_DEEP_LEARNING_GENERAL = ('epochs', 'learning_rate', 'dropout') + EXP_BATCH_ARGS_DEEP_LEARNING_GENERAL

# TODO
ARG_SETUP_NLP = ()


# endregion

class DeepLearningExperimentBase(ExperimentBase):
    def __init__(self,
                 *arg_info_objs,
                 dirs=('artifacts', 'source_data', 'datasets', 'results', 'analysis'),
                 preset_root: str = None,
                 preset: [dict, str] = None,
                 default_workspace_root='.',
                 general_args=True,
                 workspace_override_args=True,
                 deep_learning_args=True,
                 nlp_args=True,
                 expname_args=(),
                 expname_prefix='',
                 expname_suffix='',
                 batchname: str = None,
                 batchname_args=(),
                 batchname_prefix='',
                 batchname_suffix='',
                 **kwargs):
        """
        """

        arg_info_objs += ((ARG_SETUP_DEEP_LEARNING_GENERAL if deep_learning_args else ()) +
                          (ARG_SETUP_NLP if nlp_args else ()))
        expname_args += (EXP_NAME_ARGS_DEEP_LEARNING_GENERAL if deep_learning_args else ())
        batchname_args += (EXP_BATCH_ARGS_DEEP_LEARNING_GENERAL if deep_learning_args else ())
        super(DeepLearningExperimentBase).__init__(
            self,
            *arg_info_objs,
            dirs=dirs,
            preset_root=preset_root,
            preset=preset,
            default_workspace_root=default_workspace_root,
            general_args=general_args,
            workspace_override_args=workspace_override_args,
            expname_args=expname_args,
            expname_prefix=expname_prefix,
            expname_suffix=expname_suffix,
            batchname=batchname,
            batchname_args=batchname_args,
            batchname_prefix=batchname_prefix,
            batchname_suffix=batchname_suffix,
            **kwargs
        )

# class NlpExperimentBase(DeepLearningExperimentBase):
#
#     def get_vocab(self, vocab_name, build_mode=False, **kwargs) -> Vocabulary:
#         return Vocabulary(path.join(self.args.vocab_path, vocab_name), build_mode=build_mode, **kwargs)
#
#     def get_vocab__(self, vocab_config):
#         splits = vocab_config.rsplit('-', maxsplit=2)
#         if len(splits) == 1:
#             return self.get_vocab(vocab_name=vocab_config)
#         min_count = 1
#         max_size = None
#         for split in splits[1:]:
#             if split.startswith('mc'):
#                 min_count = int(split[2:])
#             elif split.startswith('ms'):
#                 max_size = int(split[2:])
#         return self.get_vocab(vocab_name=splits[0], min_count=min_count, max_size=max_size)
