def get_names():
    f = open("names2.txt", "r")
    names = []
    for line in f:
        line = line.replace(" (", ", ")
        line = line.replace(") ", ", ")
        line = line.replace(")", "")
        names.append(line)
    f.close()
    f = open("names/names3.txt", "w")
    for line in names:
        line = line.replace(", ", "\n")
        f.write(line)
    f.close()
    print(names)
    return names

def main():
    people_names = get_names()


if __name__ == "__main__":
    main()