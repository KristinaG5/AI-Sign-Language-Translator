words = {
    1: "go",
    2: "have",
    3: "football",
    4: "look",
    5: "small",
    6: "awful",
    7: "alright",
    8: "but",
    9: "good",
    10: "home",
    11: "one",
    12: "thought",
    13: "i",
    14: "before",
    15: "number",
    16: "always",
    17: "finish",
    18: "when",
    19: "five",
    20: "children",
    21: "easter",
    22: "beat",
    23: "better",
    24: "summer",
    25: "laugh",
    26: "winter",
    27: "spring",
    28: "about",
    29: "remember",
}

with open("max_warping_window_experiment.txt") as f:
    text = f.read()

for i in range(29, 0, -1):
    text = text.replace(str(i), words[i])

with open("mww_experiment_labelled.txt", "w") as f:
    f.write(text)
