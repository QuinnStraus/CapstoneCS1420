import pandas as pd
import numpy as np
from tqdm import tqdm
import re

from load_data import load_raw_data


def process_skills(df: pd.DataFrame):
    cleaned_skills = df[df["skills"].notna()]["skills"].str.replace(
        "\[|\]|\s", "", regex=True
    )
    unique_skills = cleaned_skills.str.split(",").explode().value_counts()
    advanced_skills = unique_skills[
        ~unique_skills.index.str.contains("'(?:1|2|3|4|5|6)\.", regex=True)
    ]
    unpopular_skills = advanced_skills[advanced_skills <= 50]
    unpopular_skill_regex = "|".join(
        re.escape(sk) for sk in unpopular_skills.index.to_list()
    )
    cleaned_df_skills = (
        df["skills"]
        .str.replace("\[|\]|\s", "", regex=True)
        .str.replace("'(?:1|2|3|4|5|6)\.[^']*'", "", regex=True)
        .str.replace(
            f"({unpopular_skill_regex})",
            lambda x: ".".join(x.group(0).split(".")[0:3]),
            regex=True,
        )
        .str.get_dummies(sep=",")
    )
    return cleaned_df_skills


SEQUENCE_LENGTH = 50
MIN_ENTRIES = 10
SAVE_FOLDER = "padded-50-10"


def prepare_data():
    problems, students, assignments, assignment_logs, problem_logs = load_raw_data()
    problems_onehot_skills = process_skills(problems)
    print("processed problem skills")

    categorical_data = pd.concat(
        [
            pd.get_dummies(
                problems[["problem_id", "problem_type"]],
                columns=["problem_type"],
            ),
            problems_onehot_skills,
        ],
        axis=1,
    )
    print("processed problems")
    # remove all invalid skills and convert to one-hot format

    prob_assignment_pairs = problem_logs[
        ["problem_id", "assignment_id"]
    ].drop_duplicates()
    print("problem assignment pairs", len(prob_assignment_pairs))
    probs_with_skills = pd.DataFrame(problems["problem_id"])
    probs_with_skills["has_skills"] = problems_onehot_skills.sum(axis=1) > 0

    prob_assignment_skills = pd.merge(
        prob_assignment_pairs, probs_with_skills, on="problem_id", how="inner"
    )

    # remove assignments that have only problems with no/invalid skill data
    assignment_skill_values = prob_assignment_skills.groupby("assignment_id")[
        "has_skills"
    ].any()
    valid_assignment_ids = assignment_skill_values[assignment_skill_values].index
    valid_assignment_logs = assignment_logs[
        assignment_logs["assignment_id"].isin(valid_assignment_ids)
    ]
    print("valid assignment logs", len(valid_assignment_logs))
    # remove students who have done less than 20 valid assignments
    student_assignment_counts = valid_assignment_logs["student_id"].value_counts()
    print("average sequence length", student_assignment_counts.mean())
    valid_students = students[
        students["student_id"].isin(
            student_assignment_counts[student_assignment_counts > MIN_ENTRIES].index
        )
    ]
    print(
        "valid students",
        len(valid_students),
        "average sequence length",
        student_assignment_counts[student_assignment_counts > MIN_ENTRIES].mean(),
    )
    valid_assignment_logs = valid_assignment_logs[
        valid_assignment_logs["student_id"].isin(valid_students["student_id"])
    ]
    valid_assignment_logs["start_time"] = pd.to_datetime(
        valid_assignment_logs["start_time"], format="ISO8601"
    )
    valid_assignment_logs["time_since_last"] = (
        valid_assignment_logs[["start_time", "student_id"]]
        .groupby("student_id")
        .diff()
        .fillna(pd.Timedelta(30, "d"))["start_time"]
        .dt.total_seconds()
    )
    print("usable assignment logs", valid_assignment_logs.shape)
    processed_problem_logs = problem_logs[
        problem_logs["assignment_id"].isin(valid_assignment_logs["assignment_id"])
    ]
    average_time_to_correct = (
        processed_problem_logs[processed_problem_logs["correct"] == True]
        .groupby("problem_id")[["time_on_task"]]
        .quantile(0.33)
    ).rename(columns={"time_on_task": "mean_time_on_task"})
    print(
        "average time to correct", average_time_to_correct["mean_time_on_task"].mean()
    )
    processed_problem_logs = pd.merge(
        processed_problem_logs,
        average_time_to_correct,
        on="problem_id",
        how="left",
    )
    processed_problem_logs["too_easy"] = processed_problem_logs["correct"] & (
        (
            (
                processed_problem_logs["time_on_task"]
                < (processed_problem_logs["mean_time_on_task"])
            )
        )
        | (processed_problem_logs["time_on_task"] < 20)
    )
    processed_problem_logs["good_level"] = (
        ~processed_problem_logs["too_easy"]
        & processed_problem_logs["problem_completed"]
        & (processed_problem_logs["correct"] | ~processed_problem_logs["answer_given"])
    )
    processed_problem_logs["too_hard"] = (
        ~processed_problem_logs["too_easy"] & ~processed_problem_logs["good_level"]
    )
    print("only valid problem logs", len(processed_problem_logs))
    print("too easy", processed_problem_logs["too_easy"].sum())
    print("good level", processed_problem_logs["good_level"].sum())
    print("too hard", processed_problem_logs["too_hard"].sum())
    # assign a level of 1 for hard, 0.5 for good, 0 for easy
    processed_problem_logs["difficulty"] = (
        processed_problem_logs["too_hard"].astype(int)
    ) + (processed_problem_logs["good_level"].astype(int) * 0.5)
    merged_logs = pd.merge(
        processed_problem_logs[
            ["problem_id", "assignment_id", "student_id", "difficulty"]
        ],
        categorical_data,
        on=["problem_id"],
        how="inner",
    )
    print("merged logs", len(merged_logs))
    groups = merged_logs.groupby(["assignment_id", "student_id"]).agg(
        {
            "problem_id": "size",
            **{
                col: "mean"
                for col in merged_logs.columns
                if not col in ["assignment_id", "student_id", "problem_id"]
            },
        }
    )
    groups.rename(columns={"problem_id": "num_started"}, inplace=True)
    print("grouped problem logs", groups.shape)
    assignments_with_logs = pd.merge(
        valid_assignment_logs[
            [
                "assignment_id",
                "student_id",
                "assignment_completed",
                "start_time",
                "time_since_last",
            ]
        ],
        assignments[["assignment_id", "assignment_type", "due_date"]],
        on="assignment_id",
        how="inner",
    )
    assignments_with_logs["assignment_type"] = (
        assignments_with_logs["assignment_type"] == "problem_set"
    )
    assignments_with_logs["time_until_due"] = (
        pd.to_datetime(assignments_with_logs["due_date"], format="ISO8601")
        - assignments_with_logs["start_time"]
    ).dt.total_seconds()
    print("assignments with logs", assignments_with_logs.shape)
    merged_assignment_logs = pd.merge(
        assignments_with_logs,
        groups,
        on=["assignment_id", "student_id"],
        how="inner",
    )
    print("merged assignment logs", merged_assignment_logs.shape)
    SCALAR_FEATURES = [
        "time_since_last",
        "time_until_due",
        "num_started",
        "assignment_completed",
        "assignment_type",
        "difficulty",
    ]
    # normalize scalar features\
    merged_assignment_logs[SCALAR_FEATURES] = (
        merged_assignment_logs[SCALAR_FEATURES]
        - merged_assignment_logs[SCALAR_FEATURES].mean()
    ) / merged_assignment_logs[SCALAR_FEATURES].std()
    # split data into SEQUENCE_LENGTH-assignment sequences by student
    # and move the difficulty and num_started columns to the next row
    student_assignments = merged_assignment_logs.groupby("student_id")

    # total_sequences = (student_assignments.size() - 1).floordiv(SEQUENCE_LENGTH).sum()
    total_sequences = student_assignments.ngroups
    categorical_features = [
        col
        for col in merged_assignment_logs.columns
        if col.startswith("problem_type") or col.startswith("'")
    ]
    scalar_sequences = np.ndarray(
        (total_sequences, SEQUENCE_LENGTH, len(SCALAR_FEATURES))
    )
    categorical_sequences = np.ndarray(
        (
            total_sequences,
            SEQUENCE_LENGTH,
            len(categorical_features),
        )
    )
    labels = np.ndarray((total_sequences, SEQUENCE_LENGTH, 3))
    i = 0
    for _, student_data in tqdm(student_assignments):
        student_data = student_data.sort_values("start_time")
        student_data[["difficulty", "num_started", "assignment_completed"]] = (
            student_data[["difficulty", "num_started", "assignment_completed"]]
            .shift(-1)
            .fillna(0)
        )
        # for j in range(0, (len(student_data) - 1) // SEQUENCE_LENGTH):
        #     sequence = student_data.iloc[
        #         SEQUENCE_LENGTH * j : SEQUENCE_LENGTH * (j + 1)
        #     ]
        #     result_diff = student_data.iloc[
        #         SEQUENCE_LENGTH * j + 1 : SEQUENCE_LENGTH * (j + 1) + 1
        # ]["difficulty"]
        sequence = student_data.iloc[0 : min(SEQUENCE_LENGTH, len(student_data) - 1)]
        result_diff = student_data.iloc[
            1 : min(SEQUENCE_LENGTH + 1, len(student_data))
        ]["difficulty"]
        pad_sizes = [(0, max(SEQUENCE_LENGTH - len(sequence), 0)), (0, 0)]
        labels[i] = np.pad(
            np.array(
                [
                    (result_diff < 1 / 3).values,
                    ((1 / 3 <= result_diff) & (result_diff < 2 / 3)).values,
                    (result_diff >= 2 / 3).values,
                ]
            )
            .swapaxes(0, 1)
            .astype(int),
            pad_sizes,
        )

        scalar_sequences[i] = np.pad(sequence[SCALAR_FEATURES].values, pad_sizes)
        categorical_sequences[i] = np.pad(
            sequence[categorical_features].values, pad_sizes
        )
        i += 1

    print(i, "total sequences")
    print("scalar sequences", scalar_sequences.shape)
    print("categorical sequences", categorical_sequences.shape)
    print("labels", labels.shape)
    np.save(f"cleaned/{SAVE_FOLDER}/scalar_sequences.npy", scalar_sequences)
    np.save(f"cleaned/{SAVE_FOLDER}/categorical_sequences.npy", categorical_sequences)
    np.save(f"cleaned/{SAVE_FOLDER}/labels.npy", labels)
    print("data saved")


def load_prepared_data():
    scalar_sequences: np.ndarray = np.load(
        f"cleaned/{SAVE_FOLDER}/scalar_sequences.npy"
    )
    categorical_sequences: np.ndarray = np.load(
        f"cleaned/{SAVE_FOLDER}/categorical_sequences.npy"
    )
    labels: np.ndarray = np.load(f"cleaned/{SAVE_FOLDER}/labels.npy")
    return scalar_sequences, categorical_sequences, labels


if __name__ == "__main__":
    prepare_data()
