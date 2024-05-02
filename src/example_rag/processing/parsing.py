import logging
from pathlib import Path
from typing import Tuple

from llama_index.core import SimpleDirectoryReader

LOGGER = logging.getLogger(__name__)


def index_numeral(subchapter: str) -> Tuple[str, str]:
    numeral, text = subchapter.split('\n\n', 1)
    numeral = numeral[1:]
    return numeral, text


class Section:
    def __init__(
        self, chapter_name: str, section_numeric_index: str, section_text: str
    ) -> None:
        self.chapter_name = chapter_name
        self.section_numeric_index = section_numeric_index
        self.section_text = section_text

    def save_to_file(self, output_dir: str) -> None:
        file_path = (
            Path(output_dir) / f'{self.chapter_name}_{self.section_numeric_index}.txt'
        )
        with file_path.open('w') as f:
            f.write(self.section_text)


def parse(raw_data_dir: str, parsed_data_dir: str) -> None:
    """Parse the raw data into sections and save it to files."""
    LOGGER.info('Reading raw data...')
    reader = SimpleDirectoryReader(input_dir=raw_data_dir)
    document = reader.load_data()[0]
    document.text = document.text.replace('*', '')

    LOGGER.info('Parsing data...')
    text = document.text
    splits = text.split('\n# ')

    # Manual parsing of the annotation
    annotation = '\n'.join(splits[0:3])
    annotation_section = Section(
        chapter_name='Anotacija', section_numeric_index='I', section_text=annotation
    )
    annotation_section.save_to_file(parsed_data_dir)

    # Parsing of the chapters
    chapters = splits[3:-1]
    for chapter in chapters:
        subchapters = chapter.split('##')
        chapter_name, subchapters = subchapters[0][:-2], subchapters[1:]
        for subchapter in subchapters:
            section_numeral, section_text = index_numeral(subchapter)
            section = Section(
                chapter_name=chapter_name,
                section_numeric_index=section_numeral,
                section_text=section_text,
            )
            section.save_to_file(parsed_data_dir)
