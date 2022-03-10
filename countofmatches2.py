#count of matches 2 with words to remove included 
# %%
import string
from pathlib import Path
from typing import List

from nltk.corpus import stopwords
from pyspark.ml.feature import StopWordsRemover, Tokenizer
from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql.functions import (array, array_contains, array_except,
                                   array_intersect, arrays_overlap, coalesce,
                                   col, collect_set, concat, concat_ws,
                                   explode, lit, row_number, sequence, size,
                                   to_date, translate)
import pyspark.sql.functions as sf
from pyspark.sql.types import ArrayType

# %%

DATA_FOLDER = Path('/content/drive/MyDrive/nlp').resolve()
RESULT_FOLDER = Path('/content/drive/MyDrive/nlp/finaloutputs/countofmatches2_final').resolve()

PARTIAL_RESULT_FOLDER = RESULT_FOLDER / 'partial'
# warning - it shouldn't be in partial result folder
FINAL_RESULT_FILE = RESULT_FOLDER / 'countofmatches2.csv'

ASSIGNMENT_DATA_FILE = DATA_FOLDER / 'finaloutputs/countofmatches1&1b/countofmatches_1&1b.csv'
EMPLOYEE_MAPPING_FILE = DATA_FOLDER / 'EmployeeMapping08012021.csv'
SAMPLE_STANDART_MEETING_FILE = DATA_FOLDER / 'finaloutputs/countofmatches2subjecttosubject/countofmatches2_subject_to_subject_output.csv'
CONTACT_MANAGEMENT_FILE = DATA_FOLDER / \
    'AssignmentContactManagement_09232020_10232021.csv'
MEETING_PARTICIPANTS_FILE = DATA_FOLDER / 'MeetingParticipants.csv'
PERSON_HISTORICAL_FILE = DATA_FOLDER / 'PersonHistorical.csv'
TEAM_WORKDAY_FILE = DATA_FOLDER / 'AssignmentTeamMemberWorkday_09232020_10232021.csv'

# %%
ASSIGNMENT_LIMIT = None
MEETING_LIMIT = None

# %%
# Column names
S2S_SUBJECT = 's2s_subject'
CLEANED_S2S_SUBJECT = 'cleaned_s2s_subject'
S2S_ORGANISER = 's2s_organiser'

COM_SUBJECT = 'com_subject'
CLEANED_COM_SUBJECT = 'cleaned_com_subject'
COM_ORGANISER = 'com_organiser'

# id from assignment data
ASSIGNMENT_ID = 'assignment_id'

# company name from assignment data
COMPANY_NAME = 'company_name'

# every meeting has ID
MEETING_ID = 'meeting_id'

# the start date of meeting should be between
START_DATE = 'start_date'
# booking and closed date of assignment
BOOKING_DATE = 'booking_date'
CLOSED_DATE = 'closed_date'

# assignment data has employee ids in old format
EMPLOYEE_ID_OLD = 'employee_id_old'
# we map them to new format with the help of employee mapping file
EMPLOYEE_ID_NEW = 'employee_id_new'
# name from team workday
TEAM_MEMBER_NAME = 'team_member_name_name'
# both given and family name of a meeting participant
# from contact management table
PARTICIPANT_NAME = 'participant_name'

# a list of unique organiser ids per meeting subject
UNIQUE_ORGANIZERS = 'unique_organizers'
# a list of unique team member ids per assignment id
UNIQUE_EMPLOYEE_IDS = 'unique_employees'
# a list of unique team member names per assignment id
UNIQUE_TEAM_MEMBER_NAMES = 'unique_team_member_names'
# a list of unique participant names per assignment id
UNIQUE_PART_NAMES = 'unique_part_names'
# combined
UNIQ_PART_AND_EMP_NAMES = 'unique_part_and_emp_names'
# unique name tokens
CLEANED_UNIQ_PART_AND_EMP_NAMES = f'cleaned_{UNIQ_PART_AND_EMP_NAMES}'

# assigned based on meeting id
# historical IDs are taken from meeting participants
# and matched to ids from person historical
UNIQUE_PARTICIPANT_IDS = 'unique_participant_ids'

# Historical person ID
# from meeting participants
HISTORICAL_ID = 'historical_id'

# Unique employee list and unique organiser list for every given pairing should have at least one overlap
ORG_EXISTS_IN_EMPLOYEES = 'org_exists_in_empl'
# Number of occurences of team members in participatns for every given ID
PARTICIPANTS_IN_EMPLOYEES = 'part_in_empl'
# Number of time name tokens of participants are mentioned in subject
PART_AND_EMP_IN_SUBJ = 'part_and_emp_in_subj'
# common tokens between meeting and assignment subject
SUBJ_TO_SUBJ_COMMON = "subj_to_subj_common"
# matches between meeting and assignment organisers
S2S_ORG_IN_S2S_ORG = "s2s_org_in_s2s_org"

# every participant in the meeting can be found in meeting_historical
IS_INTERNAL = 'is_internal'


# %%
def main(dry_run: bool = False):
    """
    dry_run: prints first 20 entries instead of saving the full dataframe
    """
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    assignments_and_meetings = match_assignments_and_meetings(spark)
    assignments_and_meetings.printSchema()

    if dry_run:
        # do not save all data
        assignments_and_meetings.show(truncate=False)
        return

    # Store results in the dataframe
    header = assignments_and_meetings.columns
    header = ','.join(header) + '\n'
    assignments_and_meetings.write.csv(str(PARTIAL_RESULT_FOLDER),
                                       mode='overwrite',
                                       header=False)
    print(f"Partial results are saved in {PARTIAL_RESULT_FOLDER}")

    with FINAL_RESULT_FILE.open(mode='w') as f:
        f.write(header)
        for csv_filepath in PARTIAL_RESULT_FOLDER.glob('*.csv'):
            with open(csv_filepath) as result:
                for line in result:
                    f.write(line)
                f.write(result.read())
    print(f"Results are merged in {FINAL_RESULT_FILE}")


# %%
def match_assignments_and_meetings(spark: SparkSession) -> DataFrame:
    # loading and preprocessing data
    # Constructing our final dataframe
    assignments = load_assignment(spark)
    meetings = load_sample_standard_meeting(spark)
    contact_management = load_contact_management(spark)
    meeting_participants = load_meeting_participants(spark)
    person_historical = load_person_historical(spark)
    workday = load_team_workday(spark)

    meetings = add_unique_organisers(meetings)
    meetings = add_unique_participant_ids(meetings, meeting_participants,
                                          person_historical)

    assignments = add_unique_employees(assignments, workday)
    participant_names = get_contact_management_names(contact_management)
    team_member_names = get_team_member_names(workday)
    assignments = add_unique_cleaned_names(assignments,
                                           participants=participant_names,
                                           team_members=team_member_names)

    df_master = add_meetings(assignments, meetings)
    df_master = select_best_match(df_master)

    # array columns shouldn't end up in csv files
    array_cols: list[str] = [
        c.name for c in df_master.schema  # type:ignore
        if isinstance(c.dataType, ArrayType)
    ]
    for colname in array_cols:
        df_master = df_master.withColumn(colname, format_array(col(colname)))

    final_columns = [
        col(ASSIGNMENT_ID),
        col(COMPANY_NAME),
        col(BOOKING_DATE),
        col(CLOSED_DATE),
        col(S2S_ORGANISER),
        col(COM_ORGANISER),
        col(START_DATE),
        col(S2S_SUBJECT),
        col(CLEANED_S2S_SUBJECT),
        col(COM_SUBJECT),
        col(CLEANED_COM_SUBJECT),
        col(UNIQUE_ORGANIZERS),
        col(UNIQUE_EMPLOYEE_IDS),
        col(IS_INTERNAL),
        col(ORG_EXISTS_IN_EMPLOYEES),
        col(SUBJ_TO_SUBJ_COMMON),
        col(PARTICIPANTS_IN_EMPLOYEES),
        col(PART_AND_EMP_IN_SUBJ),
        col(S2S_ORG_IN_S2S_ORG),
    ]
    df_master = df_master.select(final_columns)
    return df_master


# %%
def load_assignment(spark: SparkSession) -> DataFrame:
    df = spark.read.csv(str(ASSIGNMENT_DATA_FILE),
                        header=True,
                        inferSchema=True)
    if ASSIGNMENT_LIMIT:
        df = df.limit(ASSIGNMENT_LIMIT)
    # Renaming some columns to appropriate names
    # if the column name contains dots, I should use backticks to not treat it as struct
    df = df.select(
        col('Assignment_ID').alias(ASSIGNMENT_ID),
        col('Company_Name').alias(COMPANY_NAME),
        col('Booking_Date').alias(BOOKING_DATE),
        col('Closed_Date').alias(CLOSED_DATE),
        col('meeting_subject').alias(COM_SUBJECT),
        col('meeting_organiser').alias(COM_ORGANISER),
        col('cleaned_company_name'),
    )

    # inside the file dates are in the following format
    # 9/21/2021
    # pyspark date format specification
    # https://spark.apache.org/docs/latest/sql-ref-datetime-pattern.html
    date_format = "M/d/yyyy"
    # convert them to datetime
    df = df.withColumn(BOOKING_DATE,
                       to_date(col(BOOKING_DATE), format=date_format))
    df = df.withColumn(CLOSED_DATE,
                       to_date(col(CLOSED_DATE), format=date_format))

    if df_has_nulls(df, BOOKING_DATE):
        raise ValueError("Incorrect booking date format in assigment")
    if df_has_nulls(df, CLOSED_DATE):
        raise ValueError("Incorrect closed date format in assigment")
    # employee IDs are already converted

    df = df.distinct()
    # we need to tokenise the company name to find the common words between subject and company name
    df = text_preprocess(df, COM_SUBJECT)
    return df


def load_sample_standard_meeting(spark: SparkSession) -> DataFrame:
    # Loading the file SampleStandardMeetingQuery0820-0821.csv into a dataframe
    df = spark.read.csv(str(SAMPLE_STANDART_MEETING_FILE),
                        header=True,
                        inferSchema=True)
    if MEETING_LIMIT:
        df = df.limit(MEETING_LIMIT)
    df = df.repartition('StartDate')
    # Extracting the required columns from the dataframe
    df = df.select([
        col('StartDate').alias(START_DATE),
        col('Organizer_New_Employee_ID').alias(S2S_ORGANISER),
        col('Subject').alias(S2S_SUBJECT),
        col('MeetingId').alias(MEETING_ID)
    ])
    # Dropping/deleting the rows with null values or no values
    # Should we delete these rows or fill with "missing"?
    df = df.dropna(how='any')
    # values look like 7/26/2021
    date_format = "M/d/yyyy"
    df = df.withColumn(START_DATE, to_date(col(START_DATE),
                                           format=date_format))
    if df_has_nulls(df, START_DATE):
        raise ValueError("Incorrect start date format in meeting")

    df = df.distinct()

    # tokenising subject so we can compare it with company name
    df = text_preprocess(df, S2S_SUBJECT)
    return df


def load_contact_management(spark: SparkSession) -> DataFrame:
    df = spark.read.csv(str(CONTACT_MANAGEMENT_FILE),
                        header=True,
                        inferSchema=True)
    df = df.repartition('AssignmentId')
    df = df.select(
        col('AssignmentId').alias(ASSIGNMENT_ID),
        col('GivenNames'),
        col('FamilyNames'),
    )
    df = df.withColumn(PARTICIPANT_NAME,
                       concat_ws(' ', 'GivenNames', 'FamilyNames'))
    df = df.drop('GivenNames', 'FamilyNames')
    df = df.distinct()
    return df


def load_meeting_participants(spark: SparkSession) -> DataFrame:
    df = spark.read.csv(str(MEETING_PARTICIPANTS_FILE),
                        header=True,
                        inferSchema=True)
    df = df.select(
        col('MeetingID').alias(MEETING_ID),
        col('PersonHistoricalId').alias(HISTORICAL_ID),
    )
    return df


def load_person_historical(spark: SparkSession) -> DataFrame:
    df = spark.read.csv(str(PERSON_HISTORICAL_FILE),
                        header=True,
                        inferSchema=True)
    df = df.select(
        col('PersonHistoricalId').alias(HISTORICAL_ID),
        col('New_Employee_ID').alias(EMPLOYEE_ID_NEW))
    df = df.dropna()
    return df


def load_team_workday(spark: SparkSession) -> DataFrame:
    df = spark.read.csv(str(TEAM_WORKDAY_FILE), header=True, inferSchema=True)
    df = df.select(
        # pyspark treats columns with different case as duplicates
        # we need the second one
        col('AssignmentId48').alias(ASSIGNMENT_ID),
        col('Preferred_Name').alias(TEAM_MEMBER_NAME),
        col('Employee_ID').alias(EMPLOYEE_ID_OLD),
    )

    employee_mapping = spark.read.csv(str(EMPLOYEE_MAPPING_FILE),
                                      header=True,
                                      inferSchema=True)

    employee_mapping = employee_mapping.select(
        col('Employee_ID').alias(EMPLOYEE_ID_OLD),
        col('New_Employee_ID').alias(EMPLOYEE_ID_NEW),
    )
    df = df.join(employee_mapping, on=EMPLOYEE_ID_OLD)
    df = df.drop(EMPLOYEE_ID_OLD)

    df = df.repartition(ASSIGNMENT_ID)
    df = df.dropna()
    return df


# %%
def add_unique_organisers(meetings: DataFrame) -> DataFrame:
    """
    For every meeting subject, adds a list of unique organisers.
    Based on sample standard meeting
    """
    """
    For every meeting subject, adds a list of unique organisers
    """
    window = Window.partitionBy(S2S_SUBJECT)
    meetings = meetings.withColumn(
        UNIQUE_ORGANIZERS,
        collect_set(col(S2S_ORGANISER)).over(window))
    return meetings


def add_unique_employees(assignments: DataFrame,
                         workday: DataFrame) -> DataFrame:
    """
    For every assignment ID, get all unique IDs from workday.
    """
    team_members = workday.select(
        col(ASSIGNMENT_ID),
        col(EMPLOYEE_ID_NEW),
    )

    team_members = team_members.groupby(ASSIGNMENT_ID).agg(
        collect_set(col(EMPLOYEE_ID_NEW)).alias(UNIQUE_EMPLOYEE_IDS), )

    assignments = assignments.join(team_members, on=ASSIGNMENT_ID, how='left')

    assignments = assignments.withColumn(
        UNIQUE_EMPLOYEE_IDS,
        coalesce(col(UNIQUE_EMPLOYEE_IDS), array()),
    )
    return assignments


def get_contact_management_names(contact_management: DataFrame) -> DataFrame:
    """
    For every assigment ID, get all unique participant names
    """
    participants = contact_management.select(ASSIGNMENT_ID, PARTICIPANT_NAME)
    participants = participants.groupby(ASSIGNMENT_ID).agg(
        collect_set(col(PARTICIPANT_NAME)).alias(UNIQUE_PART_NAMES))
    return participants


def get_team_member_names(workday: DataFrame) -> DataFrame:
    """
    For every assigment ID, get all unique team member names
    """
    team_members = workday.select(ASSIGNMENT_ID, TEAM_MEMBER_NAME)
    team_members = team_members.groupby(ASSIGNMENT_ID).agg(
        collect_set(col(TEAM_MEMBER_NAME)).alias(UNIQUE_TEAM_MEMBER_NAMES))
    return team_members


def add_unique_cleaned_names(assignments: DataFrame, participants: DataFrame,
                             team_members: DataFrame) -> DataFrame:
    """
    Gives a list of unique participant name tokens per assignment id
    """
    # we get a list of unique names for every assignment id
    # we already did it in previous steps, so I'm going to reuse it
    team_members = team_members.select(
        col(ASSIGNMENT_ID),
        col(UNIQUE_TEAM_MEMBER_NAMES).alias('names'),
    )

    participants = participants.select(
        col(ASSIGNMENT_ID),
        col(UNIQUE_PART_NAMES).alias('names'),
    )

    # combine them in single table, one row per name
    # instead of combiding, we only do participants here
    combined = team_members.unionByName(participants)
    combined = combined.withColumn('names', explode(col('names')))
    # process names into tokens and give a row per token
    combined = text_preprocess(combined, 'names')
    combined = combined.withColumn('cleaned_names', explode('cleaned_names'))
    # get a list of unique tokens per assignment ID
    combined = combined.groupby(ASSIGNMENT_ID) \
        .agg(collect_set(col('cleaned_names')).alias(CLEANED_UNIQ_PART_AND_EMP_NAMES))

    # add it to the main dataframe
    assignments = assignments.join(combined, on=ASSIGNMENT_ID, how='left')
    assignments = assignments.withColumn(
        CLEANED_UNIQ_PART_AND_EMP_NAMES,
        coalesce(col(CLEANED_UNIQ_PART_AND_EMP_NAMES), array()))
    return assignments


def add_unique_participant_ids(meetings: DataFrame,
                               meeting_participants: DataFrame,
                               person_historical: DataFrame) -> DataFrame:
    meeting_participants = meeting_participants.select(MEETING_ID,
                                                       HISTORICAL_ID)
    person_historical = person_historical.select(HISTORICAL_ID,
                                                 EMPLOYEE_ID_NEW)
    participants = meeting_participants.join(person_historical,
                                             on=HISTORICAL_ID,
                                             how='left')
    participants = participants.select(MEETING_ID, EMPLOYEE_ID_NEW)

    participants = participants.groupby(MEETING_ID) \
        .agg(
            collect_set(col(EMPLOYEE_ID_NEW)).alias(UNIQUE_PARTICIPANT_IDS),
            # checking whether all participants have IDs
            (sf.max(col(EMPLOYEE_ID_NEW).isNull().cast('int')) == 0).alias(IS_INTERNAL), # type: ignore
        )

    # if we don't have a match, we treat it as empty list
    meetings = meetings.join(participants, on=MEETING_ID, how="left")
    meetings = meetings.withColumn(
        UNIQUE_PARTICIPANT_IDS, coalesce(col(UNIQUE_PARTICIPANT_IDS), array()))
    assert UNIQUE_PARTICIPANT_IDS in meetings.columns
    return meetings


def add_meetings(assignments: DataFrame, meetings: DataFrame) -> DataFrame:
    """
    Adds all possible matches between meetings and assignments based on start, booking and closing date.
    """
    # for every assignment we need to get several possible meetings
    # it is more efficient to generate a new start row for every possible date than to explicitly check the condition
    assignments = assignments.withColumn(
        START_DATE, sequence(col(BOOKING_DATE), col(CLOSED_DATE)))
    assignments = assignments.withColumn(START_DATE, explode(col(START_DATE)))

    df_master = assignments.join(meetings, on=START_DATE, how='inner')
    return df_master


def match_subj_to_subj(df: DataFrame) -> DataFrame:
    # full matches are no longer discarded
    # Find common words in subject and company name
    # Note - duplicates are not counted
    # matches from these words should not be taken in consideration
    exceptions = [
        "spencer",
        "stuart",
        "update",
        "call",
        "catch",
        "up",
        "zoom",
        "interview",
        "candidate",
        "daily",
        "weekly",
        "monthly",
        "video",
        "training",
        "session",
        "check",
        "review",
        "touch",
        "base",
        "V/call",
        "internal",
        "regroup",
        "meeting",
        "ssi",
        "teams",
        "conference",

    ]
    # tokens are always single word in lowercase without whitespace or punctuation
    exceptions = [word.lower() for word in exceptions]

    exceptions = array([lit(word) for word in exceptions])
    filtered_com_subject = array_except(col(CLEANED_COM_SUBJECT), exceptions)
    filtered_meeting_subject = array_except(col(CLEANED_S2S_SUBJECT),
                                            exceptions)
    subj_to_subj = array_intersect(filtered_com_subject,
                                   filtered_meeting_subject)
    df = df.withColumn(SUBJ_TO_SUBJ_COMMON, size(subj_to_subj))
    return df


# %%
def select_best_match(df: DataFrame) -> DataFrame:
    """
    Between pairings of assignments and meetings, select ones that meet these criteria:
    - Have >1 match between S2S subject and COM subject
    - COM organiser is in the list of S2S organisers for given S2S subject
    - There should be >1 match between employees and S2S organisers for given S2S subject
        - If not, there should be at >1 match between employees and participants

    Then, for each group of matches, baesd on combination of S2S subject and meeting start date,
    select best one.
    It is based on, by priority, first to last.
        1. Most matches between S2S subject and COM subject
        2. Most matches between participant names and S2S subject
        3. Most matches between participant and employee lists of IDs.
        4. Earliest booking date.
    """
    df = match_subj_to_subj(df)
    # only those that have overlap in subject are counted
    df = df.filter(col(SUBJ_TO_SUBJ_COMMON) > 0)

    # should match every time, but we still check it for clarity
    s2s_in_s2s = array_contains(col(UNIQUE_ORGANIZERS),
                                value=col(S2S_ORGANISER))
    df = df.withColumn(S2S_ORG_IN_S2S_ORG, s2s_in_s2s)
    # it needs to be true
    df = df.filter(S2S_ORG_IN_S2S_ORG)

    # Count of name tokens from team workday and contact management for assignment ID
    # that occur in meeting subject
    common_name_tokens = array_intersect(col(CLEANED_S2S_SUBJECT),
                                         col(CLEANED_UNIQ_PART_AND_EMP_NAMES))
    df = df.withColumn(PART_AND_EMP_IN_SUBJ, size(common_name_tokens))

    # Count of matches between employee IDs and meeting participant IDs
    common_ids = array_intersect(col(UNIQUE_EMPLOYEE_IDS),
                                 col(UNIQUE_PARTICIPANT_IDS))
    df = df.withColumn(PARTICIPANTS_IN_EMPLOYEES, size(common_ids))

    # If organisers and employees do not overlap,
    # We check that meeting participants overlap with employees
    org_in_employees = arrays_overlap(col(UNIQUE_ORGANIZERS),
                                      col(UNIQUE_EMPLOYEE_IDS))
    df = df.withColumn(ORG_EXISTS_IN_EMPLOYEES, org_in_employees)
    df = df.filter(
        col(ORG_EXISTS_IN_EMPLOYEES) | (col(PARTICIPANTS_IN_EMPLOYEES) > 0))

    window = Window.partitionBy([col(S2S_SUBJECT), col(START_DATE)]) \
        .orderBy(
        col(SUBJ_TO_SUBJ_COMMON).desc(),
        col(PART_AND_EMP_IN_SUBJ).desc(),
        col(PARTICIPANTS_IN_EMPLOYEES).desc(),
        col(BOOKING_DATE),
    )
    # Takes only the best match
    df = df.withColumn(
        'rank',
        row_number().over(window)).filter(col('rank') == 1).drop('rank')
    return df


# %%
def text_preprocess(df: DataFrame, colname: str) -> DataFrame:
    """
    Transforms a string in an array of lowercase words without punctuation
    """
    # slightly modified code from tokeniser
    df = filter_chars(df, colname)
    df = tokenise(df, colname)

    stop = stopwords.words('english')
    stop.append('hs')
    # removes extra whitespace and empty lines after tokenising
    stop.append(' ')
    stop.append('')

    df = remove_stop_words(df, stop, colname)
    return df


def filter_chars(df: DataFrame, colname: str) -> DataFrame:
    # removes punctuation
    # removes numbers
    removed = string.punctuation + string.digits
    df = df.withColumn(f'{colname}_clean', translate(colname, removed, ''))
    return df


def tokenise(df: DataFrame, colname: str) -> DataFrame:
    """
    Makes lowercase and splits by whitespace
    """
    # tokeniser can't work with null values
    df = df.dropna(subset=f'{colname}_clean')
    tokeniser = Tokenizer(inputCol=f'{colname}_clean',
                          outputCol=f'{colname}_tokens')
    df = tokeniser.transform(df)
    df = df.drop(f'{colname}_clean')
    return df


def remove_stop_words(df: DataFrame, stop: List[str],
                      colname: str) -> DataFrame:
    """
    Removes stop words and 'hs' string
    """
    remover = StopWordsRemover(inputCol=f'{colname}_tokens',
                               outputCol=f'cleaned_{colname}',
                               stopWords=stop)
    df = remover.transform(df)
    df = df.drop(f'{colname}_tokens')
    return df


# %%
# Utility functions
def format_array(array_column: Column) -> Column:
    """
    Returns the contents of array column as comma-delimited string in square brackets
    """
    return concat(lit('['), concat_ws(', ', array_column), lit(']'))


def df_has_nulls(df: DataFrame, colname: str) -> bool:
    """
    Whether the column in dataframe has nulls
    """
    # empty dataframe return None, else returns a single row
    head = df.select(colname).filter(col(colname).isNull()).head()
    assert head != []
    return head is not None


# %%
if __name__ == '__main__':
    main(dry_run=False)