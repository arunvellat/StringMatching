
from difflib import SequenceMatcher
import utils



class Ratio():
    """Computes Fuzzy Wuzzy ratio similarity measure.
      References:
    * https://pypi.python.org/pypi/fuzzywuzzy
    """

    def __init__(self):
        pass

    def get_raw_score(self, string1, string2):
        """
        Computes the Fuzzy Wuzzy ratio measure raw score between two strings.
        This score is in the range [0,100].
        Args:
            string1,string2 (str): Input strings
        Returns:
            Ratio measure raw score (int) is returned

        References:
            * https://pypi.python.org/pypi/fuzzywuzzy
        """
        # input validations
        utils.sim_check_for_none(string1, string2)
        utils.sim_check_for_string_inputs(string1, string2)

        # if one of the strings is empty return 0
        if utils.sim_check_for_empty(string1, string2):
            return 0

        string1 = utils.convert_to_unicode(string1)
        string2 = utils.convert_to_unicode(string2)

        sm = SequenceMatcher(None, string1, string2)
        return int(round(100 * sm.ratio()))

    def get_sim_score(self, string1, string2):
        """
        Computes the Fuzzy Wuzzy ratio similarity score between two strings.
        This score is in the range [0,1].
        Args:
            string1,string2 (str): Input strings
        Returns:
            Ratio measure similarity score (float) is returned
        References:
            * https://pypi.python.org/pypi/fuzzywuzzy
        """
        # input validations
        utils.sim_check_for_none(string1, string2)
        utils.sim_check_for_string_inputs(string1, string2)

        # if one of the strings is empty return 0
        if utils.sim_check_for_empty(string1, string2):
            return 0

        raw_score = 1.0 * self.get_raw_score(string1, string2)
        sim_score = raw_score / 100
        return sim_score

class TokenSort():
    """Computes Fuzzy Wuzzy token sort similarity measure.
    """

    def __init__(self):
        pass

    def _process_string_and_sort(self, s, force_ascii, full_process=True):
        """Returns a string with tokens sorted. Processes the string if
        full_process flag is enabled. If force_ascii flag is enabled then
        processing removes non ascii characters from the string."""
        # pull tokens
        ts = utils.process_string(s, force_ascii=force_ascii) if full_process else s
        tokens = ts.split()

        # sort tokens and join
        sorted_string = u" ".join(sorted(tokens))
        return sorted_string.strip()

    def get_raw_score(self, string1, string2, force_ascii=True, full_process=True):
        """
        Computes the Fuzzy Wuzzy token sort measure raw score between two strings.
        This score is in the range [0,100].
        Args:
            string1,string2 (str), : Input strings
            force_ascii (boolean) : Flag to remove non-ascii characters or not
            full_process (boolean) : Flag to process the string or not. Processing includes
            removing non alphanumeric characters, converting string to lower case and
            removing leading and trailing whitespaces.
        Returns:
            Token Sort measure raw score (int) is returned
        References:
            * https://pypi.python.org/pypi/fuzzywuzzy
        """
        # input validations
        utils.sim_check_for_none(string1, string2)
        utils.sim_check_for_string_inputs(string1, string2)

        # if one of the strings is empty return 0
        if utils.sim_check_for_empty(string1, string2):
            return 0

        sorted1 = self._process_string_and_sort(string1, force_ascii, full_process=full_process)
        sorted2 = self._process_string_and_sort(string2, force_ascii, full_process=full_process)
        ratio = Ratio()
        return ratio.get_raw_score(sorted1, sorted2)

    def get_sim_score(self, string1, string2, force_ascii=True, full_process=True):
        """
        Computes the Fuzzy Wuzzy token sort similarity score between two strings.
        This score is in the range [0,1].
        Args:
            string1,string2 (str), : Input strings
            force_ascii (boolean) : Flag to remove non-ascii characters or not
            full_process (boolean) : Flag to process the string or not. Processing includes
            removing non alphanumeric characters, converting string to lower case and
            removing leading and trailing whitespaces.
        Returns:
            Token Sort measure similarity score (float) is returned
        References:
            * https://pypi.python.org/pypi/fuzzywuzzy
        """
        raw_score = 1.0 * self.get_raw_score(string1, string2, force_ascii, full_process)
        sim_score = raw_score / 100
        return sim_score