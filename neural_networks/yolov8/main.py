from pathlib import Path

base = Path(r"C:\Users\ruslan\Documents\GitHub\final_qualifying_work\dataset")

for txt_name in ("Train.txt", "Validation.txt", "Test.txt"):
    txt_path = base / txt_name
    lines = txt_path.read_text().splitlines()
    fixed = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Конвертируем абсолютный путь обратно в относительный
        try:
            rel = Path(line).relative_to(base)
            fixed.append(str(rel).replace("\\", "/"))
        except ValueError:
            fixed.append(line)
    txt_path.write_text("\n".join(fixed))
    print(f"Fixed {txt_name}: {len(fixed)} lines")
    print(f"  Example: {fixed[0]}")