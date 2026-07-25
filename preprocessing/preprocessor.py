
import re
import unicodedata

from utils.logger import get_logger

logger = get_logger("Preprocessor")


class QualcPreprocessor:

    def __init__(self):
        pass

    def normalize_unicode(self, text):
        return unicodedata.normalize("NFKC", text)

    def remove_control_characters(self, text):
        return "".join(
            ch
            for ch in text
            if unicodedata.category(ch)[0] != "C" or ch in "\n\t"
        )

    def preprocess_text(self, text):
        text = self.normalize_unicode(text)
        text = self.remove_control_characters(text)

        lines = []

        for line in text.splitlines():

            line = re.sub(r"[ \t]+", " ", line)

            line = line.strip()

            if line:
                lines.append(line)

        return "\n".join(lines)

    def preprocess_code(self, text):

        text = self.normalize_unicode(text)

        text = text.replace("\r\n", "\n")

        text = text.replace("\r", "\n")

        lines = []

        for line in text.split("\n"):

            line = line.rstrip()

            lines.append(line)

        return "\n".join(lines)

    def preprocess_markdown(self, text):

        text = self.normalize_unicode(text)

        text = self.remove_control_characters(text)

        return text

    def preprocess_html(self, text):

        text = self.normalize_unicode(text)

        return text

    def preprocess_json(self, text):

        return text

    def preprocess(self, text, data_type="text"):

        if data_type == "text":
            return self.preprocess_text(text)

        if data_type == "code":
            return self.preprocess_code(text)

        if data_type == "markdown":
            return self.preprocess_markdown(text)

        if data_type == "html":
            return self.preprocess_html(text)

        if data_type == "json":
            return self.preprocess_json(text)

        return text
