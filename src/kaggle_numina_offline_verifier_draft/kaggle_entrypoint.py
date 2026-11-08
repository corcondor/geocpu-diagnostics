"""Fail-closed draft entry point; public source rows cannot run in this release."""


def main() -> None:
    # A separate probe cannot attest this kernel's actual metadata, mounts,
    # credentials, or network boundary. This draft must never run public rows.
    raise SystemExit("Public-code verification is blocked in this draft; a separately reviewed distribution is required")


if __name__ == "__main__":
    main()
