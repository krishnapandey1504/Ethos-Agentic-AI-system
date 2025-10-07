import pandas as pd
from rapidfuzz import process, fuzz
import argparse

def load_data(train_path, test_path):
    """Load train and test CSVs and strip column whitespaces"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()

    return train, test

def find_best_match(problem, train_problems, train_options):
    """
    Find the closest match from training problems using fuzzy matching.
    Returns the predicted correct option and reasoning trace.
    """
    result = process.extractOne(
        problem, train_problems, scorer=fuzz.token_sort_ratio, score_cutoff=50
    )
    
    if result is None:
        return 1, "No close match found in training set"
    
    # Safe unpacking for all RapidFuzz versions
    best_match = result[0]
    score = result[1]
    idx = train_problems.index(best_match)  # find index manually
    
    reasoning = f"Matched with training problem: '{best_match[:50]}...' (score={score})"
    predicted_option = train_options[idx]
    
    return predicted_option, reasoning

def predict(train, test):
    # Prepare training data
    train_problems = train['problem_statement'].astype(str).tolist()
    train_options = train['correct_option_number'].tolist()  # column from train.csv

    output_rows = []

    for i, row in test.iterrows():
        problem = row['problem_statement']  # keep exact text
        pred_option, reasoning = find_best_match(problem, train_problems, train_options)
        output_rows.append({
            "topic": row['topic'],
            "problem_statement": problem,
            "solution": reasoning,
            "correct option": pred_option
        })

    return pd.DataFrame(output_rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--input", required=True, help="Path to test.csv")
    parser.add_argument("--output", required=True, help="Path to save output.csv")
    args = parser.parse_args()

    train, test = load_data(args.train, args.input)
    predictions = predict(train, test)
    predictions.to_csv(args.output, index=False)
    print(f"✅ Results saved to {args.output}")

if __name__ == "__main__":
    main()
