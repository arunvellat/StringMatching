# Prefix Filter


import pandas as pd
from src.index.prefix_index import PrefixIndex
from src.utils.helper import convert_dataframe_to_array, find_output_attribute_indices, get_attrs_to_project, get_output_header_from_tables, get_output_row_from_tables, remove_redundant_attrs
from src.utils.token_ordering import gen_token_ordering_for_lists, gen_token_ordering_for_tables, order_using_token_ordering
from memprof import memprof


def get_prefix_length(num_tokens, sim_measure_type, threshold, tokenizer):
    """Computes prefix length.
    References:
        * String Similarity Joins: An Experimental Evaluation, VLDB 2014.
    """

    if num_tokens == 0:
        return 0

    if sim_measure_type == 'COSINE':
        return int(num_tokens -
                   ceil(threshold * threshold * num_tokens) + 1)
    elif sim_measure_type == 'EDIT_DISTANCE':
        return min(tokenizer.qval * threshold + 1, num_tokens)
    elif sim_measure_type == 'JACCARD':
        return int(num_tokens - ceil(threshold * num_tokens) + 1)
    elif sim_measure_type == 'OVERLAP':
        return max(num_tokens - threshold + 1, 0)

class PrefixFilter():
    """Finds candidate matching pairs of strings using prefix filtering
    technique.
    Args:
        tokenizer: tokenizer to be used.
        sim_measure_type (string): similarity measure type. Supported types:
            'JACCARD', 'COSINE', 'OVERLAP', 'EDIT_DISTANCE'.
        threshold (float): threshold to be used by the filter.
        allow_empty (boolean): A flag to indicate whether pairs in which both
            strings are tokenized into an empty set of tokens should
            survive the filter (defaults to True). This flag is not valid for
            measures such as 'OVERLAP' and 'EDIT_DISTANCE'.
        allow_missing (boolean): A flag to indicate whether pairs containing
            missing value should survive the filter (defaults to False).
    Attributes:
        tokenizer (Tokenizer): An attribute to store the tokenizer.
        sim_measure_type (string): An attribute to store the similarity measure
            type.
        threshold (float): An attribute to store the threshold value.
        allow_empty (boolean): An attribute to store the value of the flag
            allow_empty.
        allow_missing (boolean): An attribute to store the value of the flag
            allow_missing.
    """

    def __init__(self, tokenizer, sim_measure_type, threshold):

        sim_measure_type = sim_measure_type.upper()


        self.tokenizer = tokenizer
        self.sim_measure_type = sim_measure_type
        self.threshold = threshold



    def filter_pair(self, lstring, rstring):
        """Checks if the input strings get dropped by the prefix filter.
        Args:
            lstring,rstring (string): input strings
        Returns:
            A flag indicating whether the string pair is dropped (boolean).
        """

        # If one of the inputs is missing, then check the allow_missing flag.
        # If it is set to True, then pass the pair. Else drop the pair.
        if pd.isnull(lstring) or pd.isnull(rstring):
            return (not self.allow_missing)

        # tokenize input strings
        ltokens = self.tokenizer.tokenize(lstring)
        rtokens = self.tokenizer.tokenize(rstring)

        l_num_tokens = len(ltokens)
        r_num_tokens = len(rtokens)

        if l_num_tokens == 0 and r_num_tokens == 0:
            # if self.sim_measure_type == 'OVERLAP':
            #     return True
            # elif self.sim_measure_type == 'EDIT_DISTANCE':
            #     return False
            # else:
            #     return False
            return self.sim_measure_type == "OVERLAP"

        token_ordering = gen_token_ordering_for_lists([ltokens, rtokens])
        ordered_ltokens = order_using_token_ordering(ltokens, token_ordering)
        ordered_rtokens = order_using_token_ordering(rtokens, token_ordering)

        l_prefix_length = get_prefix_length(l_num_tokens,
                                            self.sim_measure_type,
                                            self.threshold,
                                            self.tokenizer)
        r_prefix_length = get_prefix_length(r_num_tokens,
                                            self.sim_measure_type,
                                            self.threshold,
                                            self.tokenizer)

        if l_prefix_length <= 0 or r_prefix_length <= 0:
            return True

        prefix_overlap = set(ordered_ltokens[0:l_prefix_length]).intersection(
            set(ordered_rtokens[0:r_prefix_length]))

        if len(prefix_overlap) > 0:
            return False
        else:
            return True

    def filter_tables(self, ltable, rtable,
                      l_key_attr, r_key_attr,
                      l_filter_attr, r_filter_attr,
                      l_out_attrs=None, r_out_attrs=None,
                      l_out_prefix='l_', r_out_prefix='r_'):
        """Finds candidate matching pairs of strings from the input tables using
        prefix filtering technique.
        Args:
            ltable (DataFrame): left input table.
            rtable (DataFrame): right input table.
            l_key_attr (string): key attribute in left table.
            r_key_attr (string): key attribute in right table.
            l_filter_attr (string): attribute in left table on which the filter
                should be applied.

            r_filter_attr (string): attribute in right table on which the filter
                should be applied.

            l_out_attrs (list): list of attribute names from the left table to
                be included in the output table (defaults to None).

            r_out_attrs (list): list of attribute names from the right table to
                be included in the output table (defaults to None).

            l_out_prefix (string): prefix to be used for the attribute names
                coming from the left table, in the output table
                (defaults to 'l\_').

            r_out_prefix (string): prefix to be used for the attribute names
                coming from the right table, in the output table
                (defaults to 'r\_').

        Returns:
            An output table containing tuple pairs that survive the filter
            (DataFrame).
        """


        # remove redundant attrs from output attrs.
        l_out_attrs = remove_redundant_attrs(l_out_attrs, l_key_attr)
        r_out_attrs = remove_redundant_attrs(r_out_attrs, r_key_attr)

        # get attributes to project.
        l_proj_attrs = get_attrs_to_project(l_out_attrs,
                                            l_key_attr, l_filter_attr)
        r_proj_attrs = get_attrs_to_project(r_out_attrs,
                                            r_key_attr, r_filter_attr)


        ltable_array = convert_dataframe_to_array(ltable, l_proj_attrs,
                                                  l_filter_attr)
        rtable_array = convert_dataframe_to_array(rtable, r_proj_attrs,
                                                  r_filter_attr)

        output_table = _filter_tables_split(
            ltable_array, rtable_array,
            l_proj_attrs, r_proj_attrs,
            l_key_attr, r_key_attr,
            l_filter_attr, r_filter_attr,
            self,
            l_out_attrs, r_out_attrs,
            l_out_prefix, r_out_prefix)


        # add an id column named '_id' to the output table.
        output_table.insert(0, '_id', range(0, len(output_table)))

        return output_table

    def find_candidates(self, probe_tokens, prefix_index):
        # probe prefix index to find candidates for the input probe tokens.

        if not prefix_index.index:
            return set()

        probe_num_tokens = len(probe_tokens)
        probe_prefix_length = get_prefix_length(probe_num_tokens,
                                                self.sim_measure_type,
                                                self.threshold,
                                                self.tokenizer)

        candidates = set()
        for token in probe_tokens[0:probe_prefix_length]:
            candidates.update(prefix_index.probe(token))
        return candidates


def _filter_tables_split(ltable, rtable,
                         l_columns, r_columns,
                         l_key_attr, r_key_attr,
                         l_filter_attr, r_filter_attr,
                         prefix_filter,
                         l_out_attrs, r_out_attrs,
                         l_out_prefix, r_out_prefix):
    # find column indices of key attr, filter attr and output attrs in ltable
    l_key_attr_index = l_columns.index(l_key_attr)
    l_filter_attr_index = l_columns.index(l_filter_attr)
    l_out_attrs_indices = []
    l_out_attrs_indices = find_output_attribute_indices(l_columns, l_out_attrs)

    # find column indices of key attr, filter attr and output attrs in rtable
    r_key_attr_index = r_columns.index(r_key_attr)
    r_filter_attr_index = r_columns.index(r_filter_attr)
    r_out_attrs_indices = find_output_attribute_indices(r_columns, r_out_attrs)

    # generate token ordering using tokens in l_filter_attr and r_filter_attr
    token_ordering = gen_token_ordering_for_tables(
        [ltable, rtable],
        [l_filter_attr_index, r_filter_attr_index],
        prefix_filter.tokenizer,
        prefix_filter.sim_measure_type)

    # Build prefix index on l_filter_attr
    prefix_index = PrefixIndex(ltable, l_filter_attr_index,
                               prefix_filter.tokenizer, prefix_filter.sim_measure_type,
                               prefix_filter.threshold, token_ordering)

    output_rows = []


    for r_row in rtable:
        r_string = r_row[r_filter_attr_index]

        r_filter_attr_tokens = prefix_filter.tokenizer.tokenize(r_string)
        r_ordered_tokens = order_using_token_ordering(r_filter_attr_tokens,
                                                      token_ordering)


        # probe prefix index and find candidates
        candidates = prefix_filter.find_candidates(r_ordered_tokens,
                                                   prefix_index)

        for cand in candidates:
            output_row = get_output_row_from_tables(
                ltable[cand], r_row,
                l_key_attr_index, r_key_attr_index,
                l_out_attrs_indices, r_out_attrs_indices)

            output_rows.append(output_row)



    output_header = get_output_header_from_tables(l_key_attr, r_key_attr,
                                                  l_out_attrs, r_out_attrs,
                                                  l_out_prefix, r_out_prefix)

    # generate a dataframe from the list of output rows
    output_table = pd.DataFrame(output_rows, columns=output_header)
    return output_table