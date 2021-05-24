# noxfile.py

import nox


locations = "src", "tests", "noxfile.py"


@nox.session
def lint(session):
    args = session.posargs or locations
    session.install("flake8")
    session.run("flake8", *args)


@nox.session
def tests(session):
    session.run("poetry", "install", external=True)
    session.install("pytest")
    session.run("pytest", "-v", "tests")


@nox.session
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)
