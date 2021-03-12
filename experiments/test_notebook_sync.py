from typing import List, Text
import unittest

import filecmp
import fileinput
import glob
import pathlib
import re
import subprocess
import tempfile


def _remove_preamble(lines: List[Text]) -> Text:
  preamble_matcher = re.compile(r'# ---\s*$')
  content = []

  preamble_counter = 0
  for line in lines:
    if preamble_matcher.match(line):
      preamble_counter += 1
    if preamble_counter % 2 == 0:
      content.append(line)
  return content
  

class TestJupytextNotebookSync(unittest.TestCase):
  def test_script_content_equal(self):
    fresh_tempdir = tempfile.TemporaryDirectory() 
    fresh_dir = pathlib.Path(fresh_tempdir.name)

    nb_dir = pathlib.Path(__file__).parent / 'notebooks'
    scripts_dir = pathlib.Path(__file__).parent / 'train'

    for nb_path in glob.glob('{}/*.ipynb'.format(nb_dir)):
      cur_nb = pathlib.Path(nb_path)

      cur_script = scripts_dir / '{}.py'.format(cur_nb.stem)
      new_script = fresh_dir / '{}.py'.format(cur_nb.stem)

      subprocess.run(['jupytext', 
                      '--to', 'py:percent',
                      '--output', str(new_script),
                      str(cur_nb)])

      # Test should fail because of excess metadata
      self.assertFalse(filecmp.cmp(cur_script, new_script), 
                       '{} is the same, even though it shouldn\'t'.format(
                         cur_nb.stem))

      with cur_script.open() as f:
        curr_content = _remove_preamble(f.readlines())

      with new_script.open() as f:
        new_content = _remove_preamble(f.readlines())

      self.assertListEqual(curr_content, new_content)

    fresh_tempdir.cleanup()
      

if __name__ == '__main__':
  unittest.main()