with open("debug.txt", "w") as f:
    for _ in range(1000):
        for char in ["A", "T", "C", "G"]:
            seq = char * 10000
            f.write(seq + "\n")
    