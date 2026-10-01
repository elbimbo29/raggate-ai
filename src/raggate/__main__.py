"""Allow `python -m raggate` and the `raggate` script to boot the API."""

import uvicorn


def main() -> None:
    uvicorn.run("raggate.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()