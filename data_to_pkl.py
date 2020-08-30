import pandas as pd

WINDOW_SIZE = 6
RAW_DATA_PATH = "C:\\Users\\orhai\\PycharmProjects\\SpeakersSeparator\\First_Paper\\Data\\Or_info.csv"
PICKLE_PATH = "data_to_vectors_conversion_df.pkl"

df = pd.read_csv(RAW_DATA_PATH)
df = df.dropna()

id_list = sorted(list(set(df["ID"])))[0:2]

data_to_convert_df = pd.DataFrame(columns=['ID', 'First_Word', 'Second_Word', 'Third_Word',
                                           'Fourth_Word', 'Fifth_Word', 'Sixth_Word',
                                           'First_Duration', 'Second_Duration', 'Third_Duration',
                                           'Fourth_Duration', 'Fifth_Duration', 'Sixth_Duration',
                                           'First_Normal', 'Second_Normal', 'Third_Normal',
                                           'Fourth_Normal', 'Fifth_Normal', 'Sixth_Normal',
                                           'Middle_Space', 'Label'])
file_counter = 0

for file in id_list:
    df_for_id = df[df["ID"] == file]

    for i in range(len(df_for_id) - WINDOW_SIZE):
        sub_df = df_for_id[i: i + WINDOW_SIZE]

        words = []
        durations = []
        speech_rates = []

        for j in range(WINDOW_SIZE):
            word = str(sub_df.iloc[j]["Word"])
            duration = float(sub_df.iloc[j]["To"]) - float(sub_df.iloc[j]["From"])
            sr = duration / len(word)
            words.append(word)
            durations.append(duration)
            speech_rates.append(sr)

        mid_space = float(sub_df.iloc[int(WINDOW_SIZE/2)]["From"]) - float(sub_df.iloc[int(WINDOW_SIZE/2) - 1]["To"])

        if str(sub_df.iloc[int(WINDOW_SIZE/2) - 1]["Speaker"]) != str(sub_df.iloc[int(WINDOW_SIZE/2)]["Speaker"]):
            label = "Split"
        else:
            label = "Same"

        features = [file]
        features += (words + durations + speech_rates)
        features = features + [mid_space, label]

        example_df = pd.DataFrame([features], columns=data_to_convert_df.columns.values)

        data_to_convert_df = data_to_convert_df.append(example_df)

    file_counter += 1
    print(str(file_counter) + " Out of: " + str(len(id_list)))

pd.to_pickle(data_to_convert_df, PICKLE_PATH)
