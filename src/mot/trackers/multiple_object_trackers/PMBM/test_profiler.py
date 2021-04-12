import re
from profiler import Profiler

if __name__ == "__main__":  # pragma: no cover

    with Profiler(enabled=True, contextstr="test") as p:
        for i in range(1000):
            r = re.compile(r"^$")

    print(p.get_profile_data())

    try:
        with Profiler(enabled=True, contextstr="exception") as p:
            raise ValueError("Error")
    except ValueError:
        print(p.get_profile_data())

    profiling_enabled = False
    with Profiler(enabled=profiling_enabled, contextstr="not enabled") as p:
        for i in range(1000):
            r = re.compile(r"^$")

    print(p.get_profile_data())
