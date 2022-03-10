# %%
import string
from pathlib import Path
from typing import List

from nltk.corpus import stopwords
from pyspark.ml.feature import StopWordsRemover, Tokenizer
from pyspark.sql import SparkSession, DataFrame, Window, Column
from pyspark.sql.functions import col, lit, to_date, collect_set, arrays_overlap, array_intersect, size, \
    row_number, translate, sequence, explode, concat, concat_ws, coalesce, array
from pyspark.sql.types import ArrayType

# %%
DATA_FOLDER = '/content/drive/MyDrive/nlp/'
RESULT_FOLDER = '/content/drive/MyDrive/nlp/countofmatchespysparkresults1/'

PARTIAL_RESULT_FOLDER = RESULT_FOLDER + './partial'
# warning - it shouldn't be in partial result folder
FINAL_RESULT_FILE = RESULT_FOLDER + 'matches_full.csv'

ASSIGNMENT_DATA_FILE = DATA_FOLDER + 'AssignmentTeamMemberWorkday_09232020_10232021.csv'
EMPLOYEE_MAPPING_FILE = DATA_FOLDER + 'EmployeeMapping08012021.csv'
MEETING_DATA_FILE = DATA_FOLDER + 'Standard meeting query09-23-20_10-23-21.csv'
CONTACT_MANAGEMENT_FILE = DATA_FOLDER + 'AssignmentContactManagement_09232020_10232021.csv'
MEETING_PARTICIPANTS_FILE = DATA_FOLDER + 'MeetingParticipants.csv'
PERSON_HISTORICAL_FILE = DATA_FOLDER + 'PersonHistorical.csv'

ASSIGNMENT_LIMIT = None
MEETING_LIMIT = None

# %%
# Column names
# id from assignment data
ASSIGNMENT_ID = 'assignment_id'
# organiser employee id
MEETING_ORGANISER = 'meeting_organiser'

# company name from assignment data
COMPANY_NAME = 'company_name'
# tokenised company name
CLEANED_COMPANY_NAME = f'cleaned_{COMPANY_NAME}'

# every meeting has ID
MEETING_ID = 'meeting_id'
# meeting subject from meeting data
MEETING_SUBJECT = 'meeting_subject'
# tokenised meeting subject
CLEANED_MEETING_SUBJECT = f'cleaned_{MEETING_SUBJECT}'

# the start date of meeting should be between
START_DATE = 'start_date'
# booking and closed date of assignment
BOOKING_DATE = 'booking_date'
CLOSED_DATE = 'closed_date'

# assignment data has employee ids in old format
EMPLOYEE_ID_OLD = 'employee_id_old'
# we map them to new format with the help of employee mapping file
EMPLOYEE_ID_NEW = 'employee_id_new'
# and the name of said employee
EMPLOYEE_NAME = 'employee_name'
# both given and family name of a meeting participant
# from contact management table
PARTICIPANT_NAME = 'participant_name'

# a list of unique organiser ids per meeting subject
UNIQUE_ORGANIZERS = 'unique_organizers'
# a list of unique team member ids per assignment id
UNIQUE_EMPLOYEE_IDS = 'unique_employees'
# a list of unique employee names per assignment id
UNIQUE_EMPLOYEE_NAMES = 'unique_team_member_names'
# a list of unique participant names per assignment id
UNIQUE_PARTICIPANT_NAMES = 'unique_participant_names'

# assigned based on meeting id
# historical IDs are taken from meeting participants
# and matched to ids from person historical
UNIQUE_PARTICIPANT_IDS = 'unique_participant_ids'

# Historical person ID
# from meeting participants
HISTORICAL_ID = 'historical_id'

# tokens from columns of tokenised given and family names in contact management
# combined with tokenised names from assignment workday
# unique per assignment id
UNIQ_PART_AND_EMP_NAMES = 'uniq_part_and_emp_names'
CLEANED_UNIQUE_PART_AND_EMP_NAMES = f'cleaned_{UNIQ_PART_AND_EMP_NAMES}'

# Unique employee list and unique organiser list for every given pairing should have at least one overlap
ORG_EXISTS_IN_EMPLOYEES = 'org_exists_in_empl'
# number of common words between company name and meeting subject
COMPANY_SUBJECT_COMMON = 'company_subj_common'
# Number of occurences of team members in participatns for every given ID
PARTICIPANTS_IN_EMPLOYEES = 'part_in_empl'
# Number of time name tokens of participants are mentioned in subject
PARTICIPANTS_AND_EMPLOYEES_IN_SUBJ = 'part_and_emp_in_subj'

# every participant in the meeting can be found in meeting_historical
IS_INTERNAL = 'is_internal'


# %%
def main():
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    assignments_and_meetings = match_assignments_and_meetings(spark)
    assignments_and_meetings.printSchema()
    # Store results in the dataframe
    header = assignments_and_meetings.columns
    header = ','.join(header) + '\n'
    assignments_and_meetings.write.csv(PARTIAL_RESULT_FOLDER,
                                       mode='overwrite',
                                       header=False)
    print(f"Partial results are saved in {PARTIAL_RESULT_FOLDER}")

    with open(FINAL_RESULT_FILE, 'w') as f:
        f.write(header)
        for csv_filepath in Path(PARTIAL_RESULT_FOLDER).glob('*.csv'):
            with open(csv_filepath) as result:
                for line in result:
                    f.write(line)
                f.write(result.read())
    print(f"Results are merged in {FINAL_RESULT_FILE}")


# %%
def match_assignments_and_meetings(spark: SparkSession) -> DataFrame:
    # loading and preprocessing data
    assignments = load_assignment(spark)
    meetings = load_meeting(spark)
    contact_management = load_contact_management(spark)
    meeting_participants = load_meeting_participants(spark)
    person_historical = load_person_historical(spark)

    # we prepare data for filtering
    meetings = add_unique_organisers(meetings)
    meetings = add_unique_participant_ids(meetings, meeting_participants,
                                          person_historical)

    assignments = add_unique_employees(assignments)
    assignments = add_contact_management_names(assignments, contact_management)
    assignments = add_unique_cleaned_names(assignments)

    df_master = add_meetings(assignments, meetings)
    df_master = select_best_match(df_master)

    # array columns shouldn't end up in csv files
    array_cols = [
        c.name for c in df_master.schema  # type:ignore
        if isinstance(c.dataType, ArrayType)
    ]
    for colname in array_cols:
        df_master = df_master.withColumn(colname, format_array(col(colname)))

    final_columns = [
        # assignment fields
        col(ASSIGNMENT_ID),
        col(EMPLOYEE_ID_NEW),
        col(COMPANY_NAME),
        col(CLEANED_COMPANY_NAME),
        col(BOOKING_DATE),
        col(CLOSED_DATE),
        # meeting fields
        col(MEETING_ID),
        col(START_DATE),
        col(MEETING_ORGANISER),
        col(MEETING_SUBJECT),
        col(CLEANED_MEETING_SUBJECT),
        # aggregation columns
        col(UNIQUE_ORGANIZERS),
        col(UNIQUE_EMPLOYEE_IDS),
        col(UNIQUE_PARTICIPANT_IDS),
        col(IS_INTERNAL),
        # data file becomes too massive and it's causing job failures
        # col(UNIQUE_EMPLOYEE_NAMES),
        # col(UNIQUE_PARTICIPANT_NAMES),
        # col(CLEANED_UNIQUE_PART_AND_EMP_NAMES),
        # validation columns
        col(ORG_EXISTS_IN_EMPLOYEES),
        col(COMPANY_SUBJECT_COMMON),
        col(PARTICIPANTS_IN_EMPLOYEES),
        col(PARTICIPANTS_AND_EMPLOYEES_IN_SUBJ),
    ]

    df_master = df_master.select(final_columns)
    return df_master


# %%
def load_assignment(spark: SparkSession) -> DataFrame:
    # Loading the file AssignmentTeamMemberWorkday.csv into a pandas dataframe
    df = spark.read.csv(ASSIGNMENT_DATA_FILE, header=True, inferSchema=True)
    df = df.repartition('AssignmentId48')
    if ASSIGNMENT_LIMIT:
        df = df.limit(ASSIGNMENT_LIMIT)
    # Renaming columns to appropriate names
    df = df.select(
        col('AssignmentId48').alias(ASSIGNMENT_ID),
        col('CompanyName').alias(COMPANY_NAME),
        col('BookingDate').alias(BOOKING_DATE),
        col('ClosedDate').alias(CLOSED_DATE),
        col('Employee_ID').alias(EMPLOYEE_ID_OLD),
        col('Preferred_Name').alias(EMPLOYEE_NAME),
    )
    # if the column name contains dots, I should use backticks to not treat it as struct

    # inside the file dates are in the following format
    # 2020-06-26 00:00:00.000
    date_format = "yyyy-MM-dd HH:mm:ss.SSS"
    # convert them to datetime
    df = df.withColumn(BOOKING_DATE,
                       to_date(col(BOOKING_DATE), format=date_format))
    df = df.withColumn(CLOSED_DATE,
                       to_date(col(CLOSED_DATE), format=date_format))

    # converting the employee IDs in the new format
    employee_mapping_df = spark.read.csv(EMPLOYEE_MAPPING_FILE,
                                         header=True,
                                         inferSchema=True)
    employee_mapping_df = employee_mapping_df.select(
        col('Employee_ID').alias(EMPLOYEE_ID_OLD),
        col('New_Employee_ID').alias(EMPLOYEE_ID_NEW),
    )
    df = df.join(employee_mapping_df, on=EMPLOYEE_ID_OLD, how='inner')

    df = df.distinct()
    # we need to tokenise the company name to find the common words between subject and company name
    df = text_preprocess(df, COMPANY_NAME)
    df = text_preprocess(df, EMPLOYEE_NAME)
    return df


def load_meeting(spark: SparkSession) -> DataFrame:
    # Loading the file SampleStandardMeetingQuery0820-0821.csv into a dataframe
    df = spark.read.csv(MEETING_DATA_FILE, header=True, inferSchema=True)
    if MEETING_LIMIT:
        df = df.limit(MEETING_LIMIT)

    # since we then make windows based on start dates, it may make the process more efficient
    df = df.repartition('StartDate')

    # Extracting the required columns from the dataframe
    df = df.select(
        col('MeetingID').alias(MEETING_ID),
        col('StartDate').alias(START_DATE),
        col('Organizer_New_Employee_ID').alias(MEETING_ORGANISER),
        col('Subject').alias(MEETING_SUBJECT),
    )
    # Dropping/deleting the rows with null values or no values
    df = df.dropna(how='any')
    date_format = "M/d/yyyy"
    df = df.withColumn(START_DATE, to_date(col(START_DATE),
                                           format=date_format))

    df = df.distinct()

    # tokenising subject so we can compare it with company name
    df = text_preprocess(df, MEETING_SUBJECT)
    return df


def load_contact_management(spark: SparkSession) -> DataFrame:
    df = spark.read.csv(CONTACT_MANAGEMENT_FILE, header=True, inferSchema=True)
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
    df = spark.read.csv(MEETING_PARTICIPANTS_FILE,
                        header=True,
                        inferSchema=True)
    df = df.select(
        col('MeetingID').alias(MEETING_ID),
        col('PersonHistoricalId').alias(HISTORICAL_ID),
    )
    return df


def load_person_historical(spark: SparkSession) -> DataFrame:
    df = spark.read.csv(PERSON_HISTORICAL_FILE, header=True, inferSchema=True)
    df = df.select(
        col('PersonHistoricalId').alias(HISTORICAL_ID),
        col('New_Employee_ID').alias(EMPLOYEE_ID_NEW))
    df = df.dropna()
    return df


# %%
def add_unique_organisers(meetings: DataFrame) -> DataFrame:
    """
    For every meeting subject, adds a list of unique organisers
    """
    window = Window.partitionBy(MEETING_SUBJECT)
    meetings = meetings.withColumn(
        UNIQUE_ORGANIZERS,
        collect_set(col(MEETING_ORGANISER)).over(window))
    return meetings


def add_unique_employees(assignments: DataFrame) -> DataFrame:
    """
    For every assignment id, adds a list of unique names and ids of employees.
    """
    # employees = assignments.select(ASSIGNMENT_ID, EMPLOYEE_ID_NEW, EMPLOYEE_NAME)
    # Get unique employee list per assignment id
    window = Window.partitionBy(ASSIGNMENT_ID)
    assignments = assignments.withColumn(
        UNIQUE_EMPLOYEE_IDS,
        collect_set(col(EMPLOYEE_ID_NEW)).over(window))
    assignments = assignments.withColumn(
        UNIQUE_EMPLOYEE_NAMES,
        collect_set(col(EMPLOYEE_NAME)).over(window))
    return assignments


def add_contact_management_names(assignments, contact_management):
    """
    For every assigment ID, get all unique participant malems
    """
    participants = contact_management.select(ASSIGNMENT_ID, PARTICIPANT_NAME)
    participants = participants.groupby(ASSIGNMENT_ID).agg(
        collect_set(col(PARTICIPANT_NAME)).alias(UNIQUE_PARTICIPANT_NAMES))
    assignments = assignments.join(participants, on=ASSIGNMENT_ID, how='left')
    assignments = assignments.withColumn(
        UNIQUE_PARTICIPANT_NAMES,
        coalesce(col(UNIQUE_PARTICIPANT_NAMES), array()))
    return assignments


def add_unique_cleaned_names(assignments: DataFrame) -> DataFrame:
    """
    Gives a list of unique participant name tokens per assignment id
    """
    # we get a list of unique names for every assignment id
    # we already did it in previous steps, so I'm going to reuse it
    team_members = assignments.select(
        col(ASSIGNMENT_ID),
        col(UNIQUE_EMPLOYEE_NAMES).alias('names'))
    team_members = team_members.dropDuplicates([ASSIGNMENT_ID])

    participants = assignments.select(
        col(ASSIGNMENT_ID),
        col(UNIQUE_PARTICIPANT_NAMES).alias('names'))
    participants = participants.dropDuplicates([ASSIGNMENT_ID])

    # combine them in single table, one row per name
    combined = team_members.unionByName(participants)
    combined = combined.withColumn('names', explode(col('names')))
    # process names into tokens and give a row per token
    combined = text_preprocess(combined, 'names')
    combined = combined.withColumn('cleaned_names', explode('cleaned_names'))
    # get a list of unique tokens per assignment ID
    combined = combined.groupby(ASSIGNMENT_ID) \
        .agg(collect_set(col('cleaned_names')).alias(CLEANED_UNIQUE_PART_AND_EMP_NAMES))

    # add it to the main dataframe
    assignments = assignments.join(combined, on=ASSIGNMENT_ID)
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


"""def add_meetings(assignments: DataFrame, meetings: DataFrame) -> DataFrame:
    
    #Adds all possible matches between meetings and assignments based on start, booking and closing date.
    
    # for every assignment we need to get several possible meetings
    # it is more efficient to generate a new start row for every possible date than to explicitly check the condition
    assignments = assignments.withColumn(
        START_DATE, sequence(col(BOOKING_DATE), col(CLOSED_DATE)))
    assignments = assignments.withColumn(START_DATE, explode(col(START_DATE)))
    df_master = assignments.join(meetings, on=START_DATE, how='inner')
    return df_master"""


# %%
def select_best_match(df: DataFrame) -> DataFrame:
    # Every company can have several possible meetings
    # For the same subject and same StartDate,
    # consider the best match between subject and company name
    # then matching between subject and participant names
    # then matching between employee IDs and meeting participant IDs
    # and last the earliest Booking_Date

    # Count of common words in meeting subject and company name
    # Note - duplicates are not counted
    common_words = array_intersect(col(CLEANED_COMPANY_NAME),
                                   col(CLEANED_MEETING_SUBJECT))
    df = df.withColumn(COMPANY_SUBJECT_COMMON, size(common_words))

    # Count of name tokens from team workday and contact management for assignment ID
    # that occur in meeting subject
    common_name_tokens = array_intersect(
        col(CLEANED_MEETING_SUBJECT), col(CLEANED_UNIQUE_PART_AND_EMP_NAMES))
    df = df.withColumn(PARTICIPANTS_AND_EMPLOYEES_IN_SUBJ,
                       size(common_name_tokens))

    # Count of matches between employee IDs and meeting participant IDs
    common_names = array_intersect(col(UNIQUE_EMPLOYEE_IDS),
                                   col(UNIQUE_PARTICIPANT_IDS))
    df = df.withColumn(PARTICIPANTS_IN_EMPLOYEES, size(common_names))

    # If organisers and employees do not overlap
    # We check that meeting participants overlap with employees
    df = df.withColumn(
        ORG_EXISTS_IN_EMPLOYEES,
        arrays_overlap(col(UNIQUE_ORGANIZERS), col(UNIQUE_EMPLOYEE_IDS)))
    df = df.filter(
        col(ORG_EXISTS_IN_EMPLOYEES) | (col(PARTICIPANTS_IN_EMPLOYEES) > 0))

    window = Window.partitionBy([col(MEETING_SUBJECT), col(START_DATE)]) \
        .orderBy(
        col(COMPANY_SUBJECT_COMMON).desc(),
        col(PARTICIPANTS_AND_EMPLOYEES_IN_SUBJ).desc(),
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


def format_array(array_column: Column) -> Column:
    """
    Returns the contents of array column as comma-delimited string in square brackets
    """
    return concat(lit('['), concat_ws(', ', array_column), lit(']'))


# %%
if __name__ == '__main__':
    main()
