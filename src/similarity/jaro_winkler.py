import utils
from cython.cython_jaro_winkler import jaro_winkler


class JaroWinkler():
    """Computes Jaro-Winkler measure.
    The Jaro-Winkler measure is designed to capture cases where two strings have a low Jaro score, but share a prefix and thus are likely to match.
    Args:
        prefix_weight (float): Weight to give to the prefix (defaults to 0.1).
    Attributes:
        prefix_weight (float): An attribute to store the prefix weight.
    """

    def __init__(self, prefix_weight=0.1):
        self.prefix_weight = prefix_weight

    def get_raw_score(self, string1, string2):
        """Computes the raw Jaro-Winkler score between two strings.
        Args:
            string1,string2 (str): Input strings.
        Returns:
            Jaro-Winkler similarity score (float).
        Raises:
            TypeError : If the inputs are not strings or if one of the inputs is None.
        """

        # input validations
        utils.sim_check_for_none(string1, string2)

        # convert input to unicode.
        string1 = utils.convert_to_unicode(string1)
        string2 = utils.convert_to_unicode(string2)

        utils.tok_check_for_string_input(string1, string2)

        # if one of the strings is empty return 0
        if utils.sim_check_for_empty(string1, string2):
            return 0

        return jaro_winkler(string1, string2, self.prefix_weight)

    def get_sim_score(self, string1, string2):
        """Computes the normalized Jaro-Winkler similarity score between two strings. Simply call get_raw_score.
        Args:
            string1,string2 (str): Input strings.
        Returns:
            Normalized Jaro-Winkler similarity (float).
        Raises:
            TypeError : If the inputs are not strings or if one of the inputs is None.
        """
        return self.get_raw_score(string1, string2)

    def get_prefix_weight(self):
        """Get prefix weight.
        Returns:
            prefix weight (float).
        """
        return self.prefix_weight

    def set_prefix_weight(self, prefix_weight):
        """Set prefix weight.
        Args:
            prefix_weight (float): Weight to give to the prefix.
        """
        self.prefix_weight = prefix_weight