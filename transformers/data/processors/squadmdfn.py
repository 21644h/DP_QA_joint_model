import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from ...file_utils import is_tf_available, is_torch_available
from ...tokenization_bert import whitespace_tokenize
from ...tokenization_utils_base import PreTrainedTokenizerBase, TruncationStrategy
from ...utils import logging
from .utils import DataProcessor


# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart"}


if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

logger = logging.get_logger(__name__)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def squad_convert_example_to_features(
    example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training
):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    speakers_name = []
    utter_speaker = {} #utter->speaker
    speaker_id = {} #speaker->speaker's id
    for utter_id, utter in enumerate(example.utterances):
        if len(utter['text'].strip()) > 0:
            speakers_name.append(utter['speaker'].lower()+str(':'))
            utter_speaker[utter_id] = utter['speaker'].lower()+str(':')
            speakers_set = set(speakers_name)
    for (i,name) in enumerate(speakers_set):
        speaker_id[name] = i
    #print('utter->speaker:',utter_speaker)
    #print('speaker->speakerid:',speaker_id)
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        '''if token.lower() in speakers_set:
            if i==0:
                continue
            else:
                sub_tokens = ['[SEP]']
        else:       
            if tokenizer.__class__.__name__ in [
                "RobertaTokenizer",
                "LongformerTokenizer",
                "BartTokenizer",
                "RobertaTokenizerFast",
                "LongformerTokenizerFast",
                "BartTokenizerFast",
            ]:
                sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
            else:
                sub_tokens = tokenizer.tokenize(token)'''
        if tokenizer.__class__.__name__ in [
            "RobertaTokenizer",
            "LongformerTokenizer",
            "BartTokenizer",
            "RobertaTokenizerFast",
            "LongformerTokenizerFast",
            "BartTokenizerFast",
        ]:
            sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
            if token.lower() in speakers_set:
                if i!=0:
                    sub_tokens.insert(0,'[SEP]')
        else:
            sub_tokens = tokenizer.tokenize(token)
            if token.lower() in speakers_set:
                if i!=0:
                    sub_tokens.insert(0,'[SEP]')
        for sub_token in sub_tokens:
            # print(i)
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    #print('guid:',example.qas_id)
    #print('all_doc_tokens:', all_doc_tokens)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []

    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.max_len - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.max_len - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair
    #print('tokenizer_type:',tokenizer_type)
    #print('max_len:',tokenizer.max_len)
    #print('max_len_single_sentence:',tokenizer.max_len_single_sentence)
    #print('max_len_sentences_pair:', tokenizer.max_len_sentences_pair)
    #print('sequence_added_tokens:',sequence_added_tokens)
    span_doc_tokens = all_doc_tokens
    # tokenizer.padding_side = 'left'
    while len(spans) * doc_stride < len(all_doc_tokens):

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            # texts = truncated_query
            # pairs = span_doc_tokens
            texts = span_doc_tokens
            pairs  = truncated_query
            truncation = TruncationStrategy.ONLY_SECOND.value
        else:
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value
        # print('texts',texts)

        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            texts,
            pairs,
            truncation=truncation,
            padding=padding_strategy,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_token_type_ids=True,
        )
        #20201117
        #print('='*200)
        # print('texts:',texts,'||||pairs:',pairs)
        # print('encoded_dict[input_ids]:',encoded_dict["input_ids"])

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

            # print(encoded_dict["input_ids"].index(tokenizer.pad_token_id),last_padding_id_position + 1,len(encoded_dict["input_ids"]))
        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            # index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            index = i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
            "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    #print(spans)
    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                # if tokenizer.padding_side == "left"
                # else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[1 : 1+len(span['tokens'])] = 0
        else:
            p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    # doc_offset = len(truncated_query) + sequence_added_tokens
                    doc_offset = 0

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        # sep_position = span['input_ids'].index(tokenizer.sep_token_id)
        sep_position = []
        for i in range(len(span['input_ids'])):
            if span['input_ids'][i] == 102:
                sep_position.extend([i])
        if len(sep_position)<20:
            sep_position.extend([0]*(20-len(sep_position)))

        turn_ids = []
        for i in range(len(sep_position)):
            if sep_position[i]!=0:
                if i==0:
                    turn_ids.extend([i]*(sep_position[i]-cls_index+1))
                elif sep_position[i]!=0 and sep_position[i+1]==0:
                    turn_ids.extend([-2]*(sep_position[i]-sep_position[i-1]))
                else:
                    turn_ids.extend([i]*(sep_position[i]-sep_position[i-1]))
        if(len(turn_ids)<max_seq_length):
            turn_ids.extend([-1]*(max_seq_length-len(turn_ids)))

        speaker_ids = []
        for i in range(len(turn_ids)):
            if turn_ids[i]==-1:
                speaker_ids.extend([-1])
            elif turn_ids[i]==-2:
                speaker_ids.extend([-2])
            else:
                speaker_ids.extend([speaker_id[utter_speaker[turn_ids[i]]]])

        # print("sep_position:",sep_position)    # num_of_questions*20
        # print("ajacent_matrix:",example.adjacent_matrix)     #num_of_questions*14*14

        features.append(
            SquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                sep_position,
                turn_ids,
                speaker_ids,
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                qas_id=example.qas_id,
                adjacent_matrix=example.adjacent_matrix
            )
        )
    # print('lenfeatures:',len(features))
    # def printobj(obj):
    #     for attr in dir(obj):
    #         print('[-]',attr,':',getattr(obj,attr))
    # printobj(features[0])
    # print('feature[0]:',features[0])
    # import os
    # os._exit()
    return features


def squad_convert_example_to_features_init(tokenizer_for_convert: PreTrainedTokenizerBase):
    global tokenizer
    tokenizer = tokenizer_for_convert


def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy="max_length",
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model. It is
    model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        padding_strategy: Default to "max_length". Which padding strategy to use
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset, if 'tf': returns a tf.data.Dataset
        threads: multiple processing threads.


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """
    # Defining helper methods
    features = []

    threads = min(threads, cpu_count())
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
                disable=not tqdm_enabled,
            )
        )

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
        all_turn_ids = torch.tensor([f.turn_ids for f in features],dtype=torch.long)
        all_sep_position = torch.tensor([f.sep_position for f in features], dtype=torch.long)
        all_speaker_ids = torch.tensor([f.speaker_ids for f in features],dtype=torch.long)
        all_adjacent_matrix = torch.tensor([f.adjacent_matrix for f in features],dtype=torch.long) #
        # all_qas_id = [f.qas_id for f in features]
        # print(all_token_type_ids.max())
        # print("all_sep_position:",all_sep_position.size())    # num_of_questions*20
        # print("all_ajacent_matrix:",all_adjacent_matrix.size())     #num_of_questions*14*14

        if not is_training:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            # print("all_input_ids:",all_input_ids.size())
            # print("all_feature_index:",all_feature_index)
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask, all_turn_ids, all_sep_position, all_speaker_ids,all_adjacent_matrix,all_start_positions,all_end_positions,all_is_impossible
            )
            '''initial_dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask, all_turn_ids, all_sep_position, all_speaker_ids
            )
            initial_features = []
            for f in features:
                initial_features.append(
                    InitFeatures(
                        input_ids=f.input_ids,
                        attention_mask=f.attention_mask,
                        token_type_ids=f.token_type_ids,
                        sep_position=f.sep_position,
                        turn_ids=f.turn_ids,
                        speaker_ids=f.speaker_ids,
                        cls_index=f.cls_index,
                        p_mask=f.p_mask,
                        example_index=f.example_index,
                        unique_id=f.unique_id,
                        paragraph_len=f.paragraph_len,
                        token_is_max_context=f.token_is_max_context,
                        tokens=f.tokens,
                        token_to_orig_map=f.token_to_orig_map,
                        start_position=f.start_position,
                        end_position=f.end_position,
                        is_impossible=f.is_impossible,
                        qas_id=f.qas_id
                    )
                )'''
            return features, dataset #, initial_features, initial_dataset
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_is_impossible,
                all_turn_ids,
                all_sep_position,
                all_speaker_ids,
                all_adjacent_matrix,
            )

        return features, dataset
    elif return_dataset == "tf":
        if not is_tf_available():
            raise RuntimeError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for i, ex in enumerate(features):
                if ex.token_type_ids is None:
                    yield (
                        {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "feature_index": i,
                            "qas_id": ex.qas_id,
                        },
                        {
                            "start_positions": ex.start_position,
                            "end_positions": ex.end_position,
                            "cls_index": ex.cls_index,
                            "p_mask": ex.p_mask,
                            "is_impossible": ex.is_impossible,
                        },
                    )
                else:
                    yield (
                        {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                            "feature_index": i,
                            "qas_id": ex.qas_id,
                        },
                        {
                            "start_positions": ex.start_position,
                            "end_positions": ex.end_position,
                            "cls_index": ex.cls_index,
                            "p_mask": ex.p_mask,
                            "is_impossible": ex.is_impossible,
                        },
                    )

        # Why have we split the batch into a tuple? PyTorch just has a list of tensors.
        if "token_type_ids" in tokenizer.model_input_names:
            train_types = (
                {
                    "input_ids": tf.int32,
                    "attention_mask": tf.int32,
                    "token_type_ids": tf.int32,
                    "feature_index": tf.int64,
                    "qas_id": tf.string,
                },
                {
                    "start_positions": tf.int64,
                    "end_positions": tf.int64,
                    "cls_index": tf.int64,
                    "p_mask": tf.int32,
                    "is_impossible": tf.int32,
                },
            )

            train_shapes = (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                    "feature_index": tf.TensorShape([]),
                    "qas_id": tf.TensorShape([]),
                },
                {
                    "start_positions": tf.TensorShape([]),
                    "end_positions": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "is_impossible": tf.TensorShape([]),
                },
            )
        else:
            train_types = (
                {"input_ids": tf.int32, "attention_mask": tf.int32, "feature_index": tf.int64, "qas_id": tf.string},
                {
                    "start_positions": tf.int64,
                    "end_positions": tf.int64,
                    "cls_index": tf.int64,
                    "p_mask": tf.int32,
                    "is_impossible": tf.int32,
                },
            )

            train_shapes = (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "feature_index": tf.TensorShape([]),
                    "qas_id": tf.TensorShape([]),
                },
                {
                    "start_positions": tf.TensorShape([]),
                    "end_positions": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "is_impossible": tf.TensorShape([]),
                },
            )

        return tf.data.Dataset.from_generator(gen, train_types, train_shapes)
    else:
        return features


class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and
    version 2.0 of SQuAD, respectively.
    """
    '''Processor for Molweni.'''

    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
            answer_start = None

        return SquadExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of :class:`~transformers.data.processors.squad.SquadExample` using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from `tensorflow_datasets.load("squad")`
            evaluate: Boolean specifying if in evaluation mode or in training mode

        Returns:
            List of SquadExample

        Examples::

            # >>> import tensorflow_datasets as tfds
            # >>> dataset = tfds.load("squad")

            # >>> training_examples = get_examples_from_dataset(dataset, evaluate=False)
            # >>> evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        """

        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        for tensor_dict in tqdm(dataset):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        return examples

    def get_train_examples(self, data_dir, filename=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "train")#'20201112'

    def get_dev_examples(self, data_dir, filename=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `dev-v1.1.json` and `dev-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev")#'20201112'

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        #20201112
        if input_data['title']!='train' and is_training:
            raise ValueError("This is not train.json")
        examples = []
        #20201112
        # for entry in tqdm(input_data):
        #     title = entry["title"] #
        #     for paragraph in entry["paragraphs"]:
        #         context_text = paragraph["context"]
        #         for qa in paragraph["qas"]:
        #             qas_id = qa["id"]
        #             question_text = qa["question"]
        #             start_position_character = None
        #             answer_text = None
        #             answers = []

        #             is_impossible = qa.get("is_impossible", False)
        #             if not is_impossible:
        #                 if is_training:
        #                     answer = qa["answers"][0]
        #                     answer_text = answer["text"]
        #                     start_position_character = answer["answer_start"]
        #                 else:
        #                     answers = qa["answers"]
        input_data_dialogs = input_data['dialogues']
        for dialog in tqdm(input_data_dialogs):
            context_text = dialog['context']
            for qa in dialog['qas']:
                question_text = qa['question']
                qas_id = qa['id']
                answer_text = None
                start_position_character = None
                answers = []
                is_impossible = qa.get('is_impossible', False)
                if not is_impossible:
                    if is_training:
                        answer = qa['answers'][0]
                        answer_text = answer['text']
                        start_position_character = answer['answer_start']
                    else:
                        answer = qa['answers']

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    # title=title,
                    is_impossible=is_impossible,
                    answers=answers,
                )
                examples.append(example)
        return examples


class SquadV1Processor(SquadProcessor):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"


class SquadV2Processor(SquadProcessor):
    # train_file = "train-v2.0.json"
    # dev_file = "dev-v2.0.json"
    # 20201112
    train_file = 'train.json'
    dev_file = 'dev.json'


class SquadExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        # title, 20201112
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        # self.title = title 20201112
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]


class InitFeatures:

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        sep_position,
        turn_ids,
        speaker_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
        qas_id: str = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.sep_position = sep_position
        self.turn_ids = turn_ids
        self.speaker_ids = speaker_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.qas_id = qas_id



class SquadFeatures:
    """
    Single squad example features to be fed to a model. Those features are model-specific and can be crafted from
    :class:`~transformers.data.processors.squad.SquadExample` using the
    :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        sep_position,
        turn_ids,
        speaker_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
        adjacent_matrix,
        qas_id: str = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.sep_position = sep_position
        self.turn_ids = turn_ids
        self.speaker_ids = speaker_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.qas_id = qas_id

        self.adjacent_matrix = adjacent_matrix


class SquadResult:
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id
        # self.has_log = has_log

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits

'''20201114 Molweni processor'''
class MolweniProcessor(DataProcessor):
    """Processor for the MuTual data set."""

    def get_train_examples(self, data_dir,filename):
        with open(
            os.path.join(data_dir, filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, filename):
        with open(
            os.path.join(data_dir, filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev") # , self._create_devexamples(input_data, "dev")
    
    def get_labels(self):
        """See base class."""
        return ["0", "1"] #impossible

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        #20201112
        if input_data['title']!='train' and is_training:
            raise ValueError("This is not train.json")
        examples = []
        #20201112
        # for entry in tqdm(input_data):
        #     title = entry["title"] #
        #     for paragraph in entry["paragraphs"]:
        #         context_text = paragraph["context"]
        #         for qa in paragraph["qas"]:
        #             qas_id = qa["id"]
        #             question_text = qa["question"]
        #             start_position_character = None
        #             answer_text = None
        #             answers = []

        #             is_impossible = qa.get("is_impossible", False)
        #             if not is_impossible:
        #                 if is_training:
        #                     answer = qa["answers"][0]
        #                     answer_text = answer["text"]
        #                     start_position_character = answer["answer_start"]
        #                 else:
        #                     answers = qa["answers"]

        relation_key_pair = {'Comment': 0, 'Clarification_question': 1, 'Elaboration': 2, 'Acknowledgement': 3,
                             'Continuation': 4, 'Explanation': 5, 'Conditional': 6, 'Question-answer_pair': 7,'QAP': 7,
                             'Alternation': 8, 'Q-Elab': 9, 'Result': 10, 'Background': 11, 'Narration': 12,
                             'Correction': 13, 'Parallel': 14, 'Contrast': 15}
        relation_list = ['Comment', 'Clarification_question', 'Elaboration', 'Acknowledgement',
                             'Continuation', 'Explanation', 'Conditional', 'Question-answer_pair','QAP',
                             'Alternation', 'Q-Elab', 'Result', 'Background', 'Narration',
                             'Correction',  'Parallel', 'Contrast']

        input_data_dialogs = input_data['dialogues']
        for dialog in tqdm(input_data_dialogs):
            adjacent_matrix = 16 * np.ones((15,15))
            for relations in dialog["relations"]:
                # print(relations['type'])
                # print(relation_key_pair[relations['type']])
                if relations['type'] not in relation_list:
                    print("Warning: illegal type occurs in json file")
                else:
                    adjacent_matrix[relations['y']][relations['x']] = relation_key_pair[relations['type']]
            '''flag = False
            for i in range(adjacent_matrix.shape[0]):
                num = 0
                for j in range(adjacent_matrix.shape[1]):
                    if adjacent_matrix[i][j]!=16:
                        num = num+1
                if num>1:
                    flag = True
            if flag :
                print(adjacent_matrix)'''

            context_text = dialog['context']
            utterances = dialog['edus']
            for qa in dialog['qas']:
                question_text = qa['question']
                qas_id = qa['id']
                answer_text = None
                start_position_character = None
                answers = []
                is_impossible = qa.get('is_impossible', False)
                if not is_impossible:
                    if is_training:
                        answer = qa['answers'][0]
                        answer_text = answer['text']
                        start_position_character = answer['answer_start']
                    else:
                        answers = qa['answers']

                example = MolweniExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    utterances = utterances,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    # title=title,
                    is_impossible=is_impossible,
                    answers=answers,
                    adjacent_matrix=adjacent_matrix,
                )
                examples.append(example)
        return examples

    def _create_devexamples(self, input_data, set_type):
        is_training = set_type == "train"
        #20201112
        if input_data['title']!='train' and is_training:
            raise ValueError("This is not train.json")
        examples = []
        #20201112
        # for entry in tqdm(input_data):
        #     title = entry["title"] #
        #     for paragraph in entry["paragraphs"]:
        #         context_text = paragraph["context"]
        #         for qa in paragraph["qas"]:
        #             qas_id = qa["id"]
        #             question_text = qa["question"]
        #             start_position_character = None
        #             answer_text = None
        #             answers = []

        #             is_impossible = qa.get("is_impossible", False)
        #             if not is_impossible:
        #                 if is_training:
        #                     answer = qa["answers"][0]
        #                     answer_text = answer["text"]
        #                     start_position_character = answer["answer_start"]
        #                 else:
        #                     answers = qa["answers"]


        input_data_dialogs = input_data['dialogues']
        for dialog in tqdm(input_data_dialogs):

            context_text = dialog['context']
            utterances = dialog['edus']
            for qa in dialog['qas']:
                question_text = qa['question']
                qas_id = qa['id']
                answer_text = None
                start_position_character = None
                answers = []
                is_impossible = qa.get('is_impossible', False)
                if not is_impossible:
                    if is_training:
                        answer = qa['answers'][0]
                        answer_text = answer['text']
                        start_position_character = answer['answer_start']
                    else:
                        answers = qa['answers']

                example = MyMolweniExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    utterances = utterances,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    # title=title,
                    is_impossible=is_impossible,
                    answers=answers,
                )
                examples.append(example)
        return examples

class MyMolweniExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        utterances,
        answer_text,
        start_position_character,
        # title, 20201112
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.utterances = utterances
        self.answer_text = answer_text
        # self.title = title 20201112
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        #
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

class MolweniExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        utterances,
        answer_text,
        start_position_character,
        adjacent_matrix,
        # title, 20201112
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.utterances = utterances
        self.answer_text = answer_text
        # self.title = title 20201112
        self.is_impossible = is_impossible
        self.answers = answers
        self.adjacent_matrix = adjacent_matrix

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        #
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]