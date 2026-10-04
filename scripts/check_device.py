import platform

import torch


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main() -> None:
    device = detect_device()

    print("DeepSeti device check")
    print("--------------------")
    print("Python platform:", platform.platform())
    print("PyTorch version:", torch.__version__)
    print("Selected device:", device)

    print()
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print(
            "CUDA device:",
            torch.cuda.get_device_name(0),
        )

        print(
            "CUDA device count:",
            torch.cuda.device_count(),
        )

        properties = torch.cuda.get_device_properties(0)

        vram_gb = (
            properties.total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU VRAM: {vram_gb:.2f} GB"
        )

    print()
    print(
        "MPS available:",
        torch.backends.mps.is_available(),
    )

    # Tiny tensor test
    x = torch.randn(
        128,
        128,
        device=device,
    )

    y = x @ x.T

    print()
    print(
        "Tensor test:",
        y.shape,
    )

    print(
        "Tensor device:",
        y.device,
    )

    print("Device check passed.")


if __name__ == "__main__":
    main()