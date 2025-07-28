import os
import glob
import pandas as pd

def merge_raw_csv_files(
    raw_dir="data/raw/",
    output_path="data/merged_data.csv",
    datetime_col="datetime"
):
    """
    data/raw/ 내의 모든 CSV 파일을 datetime 기준으로 병합하여
    data/merged_data.csv로 저장합니다.

    병합할 파일 예시:
    - 가격 데이터 (price.csv)
    - 감정 점수 (sentiment_scores.csv)
    - 레짐 점수 (regime_scores.csv)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 원본 CSV 파일 목록 가져오기
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        print(f"❌ 병합할 CSV 파일이 없습니다: {raw_dir}")
        return

    print(f"📄 {len(csv_files)}개의 CSV 파일 병합을 시작합니다...")

    # CSV 파일들을 DataFrame으로 읽기
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, parse_dates=[datetime_col])
            dfs.append(df)
            print(f"   - 로드 완료: {file} ({len(df)} 행)")
        except Exception as e:
            print(f"⚠️ 파일 읽기 오류: {file} - {e}")

    if not dfs:
        print("❌ 병합할 수 있는 유효한 CSV가 없습니다.")
        return

    # datetime 기준으로 순차 병합 (outer join으로 누락 데이터도 포함)
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge_asof(
            merged_df.sort_values(datetime_col),
            df.sort_values(datetime_col),
            on=datetime_col,
            direction="forward"
        )

    # 정렬 및 중복 제거
    merged_df = merged_df.drop_duplicates(subset=[datetime_col]).sort_values(datetime_col)

    # CSV 저장
    merged_df.to_csv(output_path, index=False)
    print(f"✅ 병합 완료 → {output_path} (총 {len(merged_df)} 행)")

if __name__ == "__main__":
    merge_raw_csv_files()
