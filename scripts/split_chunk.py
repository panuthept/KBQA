if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_path", type=str, required=True)
    parser.add_argument("--max_lines", type=int, default=10000)
    args = parser.parse_args()

    index = 0
    with open(args.chunk_path, "r") as f:
        lines = []
        for line in f:
            lines.append(line)
            if len(lines) >= args.max_lines:
                with open(args.chunk_path.replace(".jsonl", f"_{index}.jsonl"), "w") as out:
                    for line in lines:
                        out.write(f"{line}\n")
                index += 1
                lines = []
        if len(lines) > 0:
            with open(args.chunk_path.replace(".jsonl", f"_{index}.jsonl"), "w") as out:
                for line in lines:
                    out.write(f"{line}\n")