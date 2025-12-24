import argparse
import os

from collections import Counter
from safetensors.torch import safe_open


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def estimate_tensor_bytes(tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def common_prefix(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


def derive_schema_prefixes(keys, max_depth: int = 4) -> Counter:
    separators = [".", "/", ":", "_"]
    counts = Counter()

    for key in keys:
        parts = [key]
        for sep in separators:
            new_parts = []
            for p in parts:
                new_parts.extend(p.split(sep))
            parts = new_parts

        parts = [p for p in parts if p]
        if not parts:
            continue

        depth = min(max_depth, len(parts))
        prefix = ".".join(parts[:depth])
        counts[prefix] += 1

    return counts


def inspect_and_write_report(
    path: str,
    limit: int = 30,
    schema_depth: int = 4,
    schema_top: int = 25,
) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    base_name = os.path.splitext(os.path.basename(path))[0]
    output_path = os.path.join(
        os.path.dirname(path),
        f"{base_name}.schematics.txt"
    )

    file_size = os.path.getsize(path)

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("Tensor Container Schematic Report\n")
        out.write("=" * 80 + "\n\n")
        out.write(f"Source file: {path}\n")
        out.write(f"File size:  {format_bytes(file_size)}\n\n")

        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            total = len(keys)

            out.write(f"Total tensors: {total}\n")
            out.write(f"Showing first {min(limit, total)} entries\n\n")

            for idx, key in enumerate(keys[:limit], start=1):
                tensor = f.get_tensor(key)

                shape = tuple(tensor.shape)
                ndim = int(tensor.ndim)
                numel = int(tensor.numel())
                dtype = str(tensor.dtype).replace("torch.", "")
                bytes_est = estimate_tensor_bytes(tensor)

                out.write(f"{idx:02d}. {key}\n")
                out.write(f"    shape:  {shape}\n")
                out.write(f"    ndim:   {ndim}\n")
                out.write(f"    numel:  {numel}\n")
                out.write(f"    dtype:  {dtype}\n")
                out.write(f"    bytes:  {format_bytes(bytes_est)}\n")
                out.write("-" * 80 + "\n")

            if total > 0:
                out.write("\nLexical Schema Summary (Purely Name-Based)\n")
                out.write("=" * 80 + "\n")
                out.write(
                    f"Grouping token depth: {schema_depth} | "
                    f"Top groups shown: {schema_top}\n\n"
                )

                counts = derive_schema_prefixes(keys, max_depth=schema_depth)
                for prefix, count in counts.most_common(schema_top):
                    out.write(f"{count:6d}  {prefix}\n")

                keys_sorted = sorted(keys)
                shared = keys_sorted[0]
                for k in keys_sorted[1:]:
                    shared = common_prefix(shared, k)
                    if not shared:
                        break

                out.write("\nCommon character prefix across all keys:\n")
                out.write(shared if shared else "(none)")
                out.write("\n")

    print(f"Report written to:\n  {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a schematic report for a .safetensors file."
    )
    parser.add_argument("path", type=str, help="Path to .safetensors file")
    parser.add_argument("--limit", type=int, default=30, help="Number of keys to list (default: 30)")
    parser.add_argument("--schema-depth", type=int, default=4, help="Token depth for schema grouping")
    parser.add_argument("--schema-top", type=int, default=25, help="Number of schema groups to show")

    args = parser.parse_args()

    inspect_and_write_report(
        path=args.path,
        limit=args.limit,
        schema_depth=args.schema_depth,
        schema_top=args.schema_top,
    )


if __name__ == "__main__":
    main()
