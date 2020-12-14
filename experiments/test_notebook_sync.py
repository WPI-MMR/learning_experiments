from logging import currentframe
import unittest

import filecmp
import fileinput
import glob
import pathlib
import re
import subprocess
import sys
import tempfile


class TestJupytextNotebookSync(unittest.TestCase):
  def test_script_content_equal(self):
    # Running jupytext from the cli adds different metadata than running it from
    # jupyter, so need to account for that.
    format_pattern = r'#\s*formats: .*'
    format_matcher = re.compile(format_pattern)

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

      # Delete the metatdata inplace
      with fileinput.FileInput(new_script, inplace=1) as new_script_fh:
        for line in new_script_fh:
          if not format_matcher.match(line):
            print(line, end='')
      
      self.assertTrue(filecmp.cmp(cur_script, new_script),
                      '{} differs! Pls ensure that everything\'s synced'.format(
                        cur_nb.stem))
    fresh_tempdir.cleanup()
      

if __name__ == '__main__':
  unittest.main()