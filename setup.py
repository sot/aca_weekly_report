# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

entry_points = {"console_scripts": ["aca_weekly_report=aca_weekly_report.report:main"]}

setup(
    name="aca_weekly_report",
    author="Jean Connelly",
    description="make weekly report on observed/tracked metrics",
    author_email="jconnelly@cfa.harvard.edu",
    packages=["aca_weekly_report"],
    package_data={"aca_weekly_report": ["*template*html", "task_schedule.cfg"]},
    include_package_data=True,
    license=(
        "New BSD/3-clause BSD License\nCopyright (c) 2019"
        " Smithsonian Astrophysical Observatory\nAll rights reserved."
    ),
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    entry_points=entry_points,
    zip_safe=False,
    tests_require=["pytest"],
    cmdclass=cmdclass,
)
