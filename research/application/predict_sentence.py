def remove_low_confidence(values, confidence, min_confidence=0.4):
    for i in range(len(values)):
        if confidence[i] < min_confidence:
            values[i] = 0
    return values


def remove_outliers(values, min_length=3):
    current = values[0]
    starting_index = 0
    total = 1
    for i in range(1, len(values)):
        if values[i] == current:
            total += 1
        else:
            if total < min_length:
                for j in range(starting_index, starting_index + total):
                    values[j] = 0
            current = values[i]
            total = 1
            starting_index = i
    return values


def label_sequence(values, num_to_class):
    current_value = 0
    starting_index = 0
    word_ranges = []
    for i in range(len(values)):
        if values[i] != 0:
            if current_value == 0:
                current_value = values[i]
            elif values[i] != current_value:
                # end sequence
                word_ranges.append({"word": current_value, "start": starting_index, "end": i})
                current_value = 0
                starting_index = i
    if current_value != 0:
        word_ranges.append({"word": current_value, "start": starting_index, "end": i})

    for word in word_ranges:
        word["word"] = num_to_class[word["word"]]
    return word_ranges


def predict_sentence(values, confidence, num_to_class):
    values = remove_low_confidence(values, confidence)
    print(values)
    values = remove_outliers(values)
    print(values)
    return label_sequence(values, num_to_class)
