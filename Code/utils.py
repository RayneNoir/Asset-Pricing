import pandas as pd

def open_file(file, percentage=False):
    """
    Prepares the dataset of returns/factors

    :param file: The name of the file containing the returns/factors
    :param percentage: If True, the input returns/factors are in percentages and are converted to decimal values, False otherwise
    :return: A pandas DataFrame containing the returns/factors for the period of interest
    """

    df = pd.read_csv(file)

    # Converting the dates as Y-M format and set it as index column
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m')
    else:
        df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce', dayfirst=True)

    df.set_index('date', inplace=True)
    df.index = pd.PeriodIndex(df.index, freq='M')
    if percentage:
        df = df/100

    # Finding the lowest student number in the group for the start and end date
    groupNames = [("Brinkhof", "Thijmen"), ("Fan", "Zhi Yang"), ("Visser", "Joeri de")]

    _, _, startDate, endDate = find_period(groupNames)

    df = df.loc[startDate:endDate]
    return df

def find_period(names):
    """
    This function searches for the lowest student number among the group members
    and for that student number it returns the start and end date of the relevant period

    :param names: the names of the group members
    :return: A dataframe containing information about the group members, the lowest student number,
    and the student number's corresponding start and end date for the period of interest
    """

    # Re-organizing the DataFrame
    df = pd.read_excel('start_end_fem21003.xlsx')
    df.rename(columns={'Unnamed: 0': 'last name', 'Unnamed: 1': 'first name', 'Unnamed: 3': 'student number'}, inplace=True)
    df['Start'] = df['Start'].dt.to_period('M')
    df['End'] = df['End'].dt.to_period('M')
    df['student number'] = df["student number"].str.extract(r'(^\d{6})')
    df["last name"] = df["last name"].str.strip()
    df["first name"] = df["first name"].str.strip()
    df.drop(columns=['Unnamed: 2'], inplace=True)

    # Create a MultiIndex from the DataFrame's two columns
    pairs = list(zip(df["last name"], df["first name"]))
    mask = pd.Series(pairs).isin(names)

    result = df.loc[mask]

    studentNumber = result['student number'].min()
    startDate = result['Start'][result['student number'] == studentNumber].squeeze()
    endDate = result['End'][result['student number'] == studentNumber].squeeze()

    return result, studentNumber, startDate, endDate

# Testing purposes
#groupNames = [("Brinkhof", "Thijmen"), ("Fan", "Zhi Yang"), ("Visser", "Joeri de")]
#result, studentNumber, startDate, endDate = find_period(groupNames)