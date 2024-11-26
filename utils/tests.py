import pytest
import sys

def main():
    errno = pytest.main()
    sys.exit(errno)


if __name__ == "__main__":
    main()