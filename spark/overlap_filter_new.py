import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
spark = SparkSession.builder.config('spark.driver.memory', '30g').config('spark.executor.memory',
                                                                        '30g').config('setMaster', 'localhost').getOrCreate()
#spark = SparkSession.builder.config('setMaster', 'localhost').getOrCreate()
from pyspark.sql.functions import udf, split, explode, monotonically_increasing_id, collect_list, row_number
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.sql.window import Window as W
from pyspark.sql import functions as F
spark_table = spark.read.csv('matchingNames.csv', header=True)
spark_table = spark_table.drop('_c0')
spark_table = spark_table.drop('Split_Aliases')
spark_table = spark_table.withColumn("index", monotonically_increasing_id())
#windowSpec = W.orderBy("index")
#spark_table = spark_table.withColumn("index", F.row_number().over(windowSpec))
spark_table = spark_table.limit(int(sys.argv[1]))

import py_stringmatching as sm
qg3_tok = sm.QgramTokenizer(qval=3, prefix_pad=" ", suffix_pad=" ")


class InvertedIndex():
    def __init__(self, key_attr, index_attr, tokenizer):
        self.key_attr = key_attr
        self.index_attr = index_attr
        self.tokenizer = tokenizer
        self.index = {}

    def build(self, table):

        '''
        generate_tokens_udf = udf(self.tokenizer, ArrayType(StringType()))
        inverted_index_df = table.withColumn('ngrams_names', generate_tokens_udf(self.index_attr)) \
            .select(*[self.key_attr, explode(df1['ngrams_names']).alias("trigram")]) \
            .groupby('trigram').agg(collect_list(self.key_attr).alias("index_list"))
        '''

        self.index = table.filter(self.index_attr + ' is not null') \
            .rdd.map(lambda row: (row[self.key_attr], set(self.tokenizer(row[self.index_attr])))) \
            .flatMapValues(lambda x: x) \
            .map(lambda x: (x[1], x[0])) \
            .groupByKey() \
            .mapValues(list) \
            .collectAsMap()
    
        return True

    def probe(self, token):
        return self.index.get(token, [])

tokenizer = qg3_tok.tokenize
l_key_attr = 'index'
l_join_attr = 'names'
comp_op = '>='

import operator
COMP_OP_MAP = {'>=': operator.ge,
               '>': operator.gt,
               '<=': operator.le,
               '<': operator.lt,
               '=': operator.eq,
               '!=': operator.ne}

def overlap(set1, set2):
    """Computes the overlap between two sets.
    Args:
        set1,set2 (set or list): Input sets (or lists). Input lists are
            converted to sets.
    Returns:
        overlap (int)
    """
    if not isinstance(set1, set):
        set1 = set(set1)
    if not isinstance(set2, set):
        set2 = set(set2)
    return len(set1.intersection(set2))

# Build inverted index over ltable
inverted_index = InvertedIndex(l_key_attr, l_join_attr, tokenizer)
inverted_index.build(spark_table)

inverted_index_bd = spark.sparkContext.broadcast(inverted_index.index)


class OverlapFilter():
    """
    Overlap filter class.
    Attributes:
        tokenizer: Tokenizer function
        sim_measure_type: String, similarity measure type.
        comp_op (string): comparison operator. '>=', '>' and '=' (defaults to '>=')
        threshold: similarity threshold to be used by the filter
    """

    def __init__(self, tokenizer, threshold, comp_op='>='):
        self.tokenizer = tokenizer
        self.comp_op = comp_op
        self.threshold = threshold
        # super(self.__class__, self).__init__()

    def filter_pair(self, lstring, rstring):
        """Filter two strings with position filter.
        Args:
        lstring, rstring : input strings
        Returns:
        result : boolean, True if the tuple pair is dropped.
        """
        # check for empty string
        if (not lstring) or (not rstring):
            return True

        ltokens = list(set(self.tokenizer(lstring)))
        rtokens = list(set(self.tokenizer(rstring)))

        num_overlap = overlap(ltokens, rtokens)

        if COMP_OP_MAP[self.comp_op](num_overlap, self.threshold):
            return False
        else:
            return True

    def _find_candidates(self, probe_tokens, inverted_index):
        candidate_overlap = {}

        if not inverted_index:
            return candidate_overlap

        for token in probe_tokens:
            for cand in inverted_index.get(token, []):
                candidate_overlap[cand] = candidate_overlap.get(cand, 0) + 1
        return candidate_overlap

of = OverlapFilter(tokenizer,1)

def apply_overlap_filter(candidates, overlap_filter, inverted_index_bd):
    """
    Apply overlap filter.
        Args:
        candidates : Spark RDD, tuples of form (r_id, r_tokens)
        overlap_filter: overlap filter
        inverted_index_bd : Inverted Index broadcast variable
        Returns:
        result : Spark RDD, tuples of form (r_id, l_id, overlap)
    """
    comp_fn = COMP_OP_MAP[overlap_filter.comp_op]
    return candidates.map(lambda pair_r: ((pair_r[0], pair_r[1]),overlap_filter._find_candidates(pair_r[1],inverted_index_bd.value))) \
            .flatMapValues(lambda x: x.items()) \
            .filter(lambda pair: comp_fn(pair[1][1], overlap_filter.threshold)) \
            .map(lambda pair: (pair[0][0], pair[1][0], pair[1][1]))

def get_table_dict(table, tokenizer, key_attr, join_attr):
    """Get table row id mapped to ordered token of joined attribute.
           Args:
           candidates : Spark dataframe, ltable or rtable
           tokenizer: tokenizer
           key_attr : String, table key attribute
           join_attr : String, table join attribute
           Returns:
           result : Spark RDD, tuples of form (id, tokens)
           """
    return table.filter(join_attr + ' is not null') \
        .rdd.map(lambda row: (row[key_attr], set(tokenizer(str(row[join_attr])))))

r_join_attr_dict = get_table_dict(spark_table, tokenizer, "index", "most_similar")

candidates = apply_overlap_filter(r_join_attr_dict, of, inverted_index_bd)

def add_unique_cand_id(candidates):
    """Add candidateIds
              Args:
              candidates : Spark RDD, tuples of form (r_id, l_id, sim_score)
              Returns:
              result : Spark RDD, tuples of form (candset_id, l_id, r_id, sim_score)
              """
    return candidates.zipWithUniqueId() \
        .map(lambda pair: (pair[1], pair[0][1], pair[0][0], pair[0][2]))


candidates = add_unique_cand_id(candidates)

def get_output_header_from_tables(candset_key_attr,
                                  l_key_attr, r_key_attr,
                                  l_out_attrs, r_out_attrs,
                                  l_out_prefix, r_out_prefix):
    output_header = []

    output_header.append(candset_key_attr)

    output_header.append(l_out_prefix + l_key_attr)

    if l_out_attrs:
        for l_attr in l_out_attrs:
            output_header.append(l_out_prefix + l_attr)

    output_header.append(r_out_prefix + r_key_attr)

    if r_out_attrs:
        for r_attr in r_out_attrs:
            output_header.append(r_out_prefix + r_attr)

    return output_header

output_header = get_output_header_from_tables('_id',
                                                  'index', 'index',
                                                  None, None,
                                                  'l_', 'r_')
out_sim_score = True
if out_sim_score:
        output_header.append("_sim_score")

result_table = spark.createDataFrame(candidates, output_header)

from pyspark.sql.functions import col
result_table.sort(col('_sim_score').desc()).show(100)

print(result_table.count())