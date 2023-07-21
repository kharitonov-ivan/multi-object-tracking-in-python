# noxfile.py

import nox


locations = "src", "tests", "noxfile.py"
nox.options.sessions = "lint", "black", "tests"


@nox.session
def lint(session):
    args = session.posargs or locations
    session.install("ruff")
    session.run("ruff", *args)


@nox.session(reuse_venv=True)
def vulture(session):
    """Find dead code"""
    session.install("vulture")
    session.run("vulture", ".")


@nox.session
def tests(session):
    session.run("poetry", "install", external=True)
    session.install("pytest")
    session.run("pytest", "-v", "tests/PMBM")


@nox.session
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)
