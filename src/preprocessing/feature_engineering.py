import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer

def clean_list(x):
    if not isinstance(x, str):
        return []

    items = re.split(r'[,-]', x)
    cleaned = []

    for i in items:
        i = i.strip()
        if not i or i == 'لا يوجد':
            continue

        # Normalize Arabic
        i = i.replace('الإنحناءات', 'الانحناءات')
        if i == 'تبسيط الرسم':
            i = 'تبسيط'
        if i == 'صعوبات تقاطع':
            i = 'صعوبات التقاطع'
        if i == 'مداومة':
            i = 'المداومة'

        cleaned.append(i)

    return cleaned


def run_feature_engineering(input_path, output_path):
    df = pd.read_excel(input_path)

    df['التغيرات في الجشطالت'] = df['التغيرات في الجشطالت'].apply(clean_list)
    df['ملاحظات دالة بشكل عام'] = df['ملاحظات دالة بشكل عام'].apply(clean_list)

    mlb_gestalt = MultiLabelBinarizer()
    gestalt_expanded = pd.DataFrame(
        mlb_gestalt.fit_transform(df['التغيرات في الجشطالت']),
        columns=[f"جشطالت_{c}" for c in mlb_gestalt.classes_],
        index=df.index
    )

    mlb_notes = MultiLabelBinarizer()
    notes_expanded = pd.DataFrame(
        mlb_notes.fit_transform(df['ملاحظات دالة بشكل عام']),
        columns=[f"ملاحظات_{c}" for c in mlb_notes.classes_],
        index=df.index
    )

    df_engineered = pd.concat([df, gestalt_expanded, notes_expanded], axis=1)

    df_engineered = df_engineered.drop(columns=[
        'التغيرات في الجشطالت',
        'ملاحظات دالة بشكل عام'
    ])

    df_engineered.to_csv(output_path, index=False)

    print("Feature engineering done!")


if __name__ == "__main__":
    input_file = "data/raw/Bander_NewTest_Responses.xlsx"
    output_file = "data/processed/engineered_features.csv"

    run_feature_engineering(input_file, output_file)