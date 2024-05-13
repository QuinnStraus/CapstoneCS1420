import pandas as pd

AUG_PROBLEM_PATH = "./data/aug_to_oct/pdets.csv"
NOV_PROBLEM_PATH = "./data/nov_to_jan/pdets.csv"
PROBLEM_LOG_PATH = "./data/plogs.pkl"
AUG_STUDENT_PATH = "./data/aug_to_oct/sdets.csv"
NOV_STUDENT_PATH = "./data/nov_to_jan/sdets.csv"
AUG_ASSIGNMENT_PATH = "./data/aug_to_oct/adets.csv"
NOV_ASSIGNMENT_PATH = "./data/nov_to_jan/adets.csv"
AUG_ASSIGNMENT_LOG_PATH = "./data/aug_to_oct/alogs.csv"
NOV_ASSIGNMENT_LOG_PATH = "./data/nov_to_jan/alogs.csv"


def pickle_problem_logs():
    aug_problem_logs = pd.read_csv("./data/aug_to_oct/plogs.csv")
    nov_problem_logs = pd.read_csv("./data/nov_to_jan/plogs.csv")
    problem_logs = pd.concat([aug_problem_logs, nov_problem_logs])
    print("loaded logs")
    problem_logs.to_pickle(PROBLEM_LOG_PATH)
    print("saved logs")


def load_raw_data():
    problems = pd.concat(
        [
            pd.read_csv(AUG_PROBLEM_PATH),
            pd.read_csv(NOV_PROBLEM_PATH),
        ]
    ).drop_duplicates("problem_id")
    print("loaded problems", problems.shape)
    students = pd.concat(
        [
            pd.read_csv(AUG_STUDENT_PATH),
            pd.read_csv(NOV_STUDENT_PATH),
        ]
    ).drop_duplicates("student_id")
    print("loaded students", students.shape)
    assignments = pd.concat(
        [
            pd.read_csv(AUG_ASSIGNMENT_PATH),
            pd.read_csv(NOV_ASSIGNMENT_PATH),
        ]
    ).drop_duplicates("assignment_id")
    print("loaded assignments", assignments.shape)
    assignment_logs = pd.concat(
        [
            pd.read_csv(AUG_ASSIGNMENT_LOG_PATH),
            pd.read_csv(NOV_ASSIGNMENT_LOG_PATH),
        ]
    )
    print("loaded assignment logs", assignment_logs.shape)
    problem_logs: pd.DataFrame = pd.read_pickle(PROBLEM_LOG_PATH)
    print("loaded problem logs", problem_logs.shape)
    return problems, students, assignments, assignment_logs, problem_logs
