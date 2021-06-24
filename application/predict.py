import numpy as np
from functools import cmp_to_key


def sort_by_start(item1, item2):
    """Sort items from start frame

    Arguments
    ---------
    item1 : int
    First start frame

    item2 : int
    Second start frame

    Returns
    ---------
    Int
    """
    return item1["start"] - item2["start"]


def sort_by_end(item1, item2):
    """Sort items from end frame

    Arguments
    ---------
    item1 : int
    First end frame

    item2 : int
    Second end frame

    Returns
    ---------
    Int
    """
    return item1["end"] - item2["end"]


def join_results(results):
    """Produce a dictionary of all of the results

    Arguments
    ---------
    results : array
    Array of predicted results

    Returns
    ---------
    Dictionary of the predictions
    """
    total_score = sum([x["score"] for x in results])
    return {"word": results[0]["word"], "start": results[0]["start"], "end": results[-1]["end"], "score": total_score}


def predict_sentence(data, num_to_class, min_confidence=0.2, min_score=1, min_separation=3):
    """Post-processing function to predict the whole sentence of a video

    Arguments
    ---------
    data : Array
    Array of predictions made by the model

    num_to_class : Array
    Array of classes 

    min_confidence : float
    Minimum confidence threshold

    min_score : int
    Minimum score threshold
    
    min_separation : int
    Lenght of how far apart each word is allowed to be

    Returns
    ---------
    Array of predictions
    """
    num_classes = len(data[0])
    results = []
    # Getting words
    for index in range(num_classes):
        word = num_to_class[index]
        start = 0
        score = 0

        for i in range(len(data)):
            if data[i][index] >= min_confidence:
                score += data[i][index]
            else:
                if score:
                    results.append({"word": word, "start": start, "end": i - 1, "score": score})
                start = i + 1
                score = 0

        if score:
            results.append({"word": word, "start": start, "end": i, "score": score})

    # Join matching words in close proximity
    joined_results = []
    for i in range(num_classes):
        matches_for_class = [x for x in results if x["word"] == num_to_class[i]]
        matches_for_class = sorted(matches_for_class, key=cmp_to_key(sort_by_end))

        start = 0
        end = 0
        if len(matches_for_class) > 1:
            for j in range(len(matches_for_class) - 1):
                seperation = matches_for_class[j + 1]["start"] - matches_for_class[j]["end"]
                if seperation <= min_separation:
                    end = j + 1
                else:
                    if start != end:
                        joined_results.append(join_results(matches_for_class[start : end + 1]))
                    else:
                        joined_results.append(matches_for_class[j])
                        # Add last if at end
                        if j == len(matches_for_class) - 2:
                            joined_results.append(matches_for_class[j + 1])
                    start = j + 1
                    end = j + 1

            if start != end:
                joined_results.append(join_results(matches_for_class[start : end + 1]))

        elif len(matches_for_class) == 1:
            joined_results.append(matches_for_class[0])

    # Removes words with low scores
    scored_results = []
    for result in joined_results:
        if result["score"] >= min_score:
            result["score"] = round(result["score"], 2)
            scored_results.append(result)

    scored_results = sorted(scored_results, key=cmp_to_key(sort_by_start))

    final_results = scored_results.copy()
    matching = True
    # Remove fully overlapping words
    while matching:
        matching = False
        for i in range(len(final_results) - 1, 0, -1):
            if final_results[i - 1]["end"] >= final_results[i]["end"]:
                matching = True
                if final_results[i - 1]["score"] >= final_results[i]["score"]:
                    final_results.pop(i)
                else:
                    final_results.pop(i - 1)

    # Fix word endings
    for i in range(1, len(final_results)):
        if final_results[i - 1]["end"] > final_results[i]["start"]:
            final_results[i - 1]["end"] = final_results[i]["start"]

    return final_results
