# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import sys
from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

if "--user" not in sys.argv:
    share_path = os.path.join(sys.prefix, "share", "aca_weekly_report")
    data_files = [(share_path, ['task_schedule.cfg'])]
else:
    data_files = None

entry_points = {'console_scripts': ['aca_weekly_report=aca_weekly_report.report:main']}

setup(name='aca_weekly_report',
      author='Jean Connelly',
      description='make weekly report on observed/tracked metrics',
      author_email='jconnelly@cfa.harvard.edu',
      packages=['aca_weekly_report'],
      package_data={'aca_weekly_report': ['*template*html']},
      license=("New BSD/3-clause BSD License\nCopyright (c) 2019"
               " Smithsonian Astrophysical Observatory\nAll rights reserved."),
      use_scm_version=True,
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      entry_points=entry_points,
      zip_safe=False,
      tests_require=['pytest'],
      cmdclass=cmdclass, install_requires=[]
      )
