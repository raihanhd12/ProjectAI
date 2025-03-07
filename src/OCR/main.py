# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import cv2
import easyocr
import pytesseract
import textdistance
import os
import re
from operator import itemgetter
import time
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill

# Define paths to data and images
DATA_DIR = "./data"
IMAGE_DIR = "./ktp"
MODULE_IMAGE_PATH = os.path.join(DATA_DIR, "module.JPEG")

# Load dataframes from CSV files
kki_prepared_df = pd.read_csv(os.path.join(DATA_DIR, "kki_prepared.csv"))
marriage_df = pd.read_csv(os.path.join(DATA_DIR, "marriage.csv"))
id_card_keyword_df = pd.read_csv(os.path.join(DATA_DIR, "id_card_keyword.csv"))
blood_df = pd.read_csv(os.path.join(DATA_DIR, "blood.csv"))
religion_df = pd.read_csv(os.path.join(DATA_DIR, "religion.csv"))

# Get list of image files
file_list = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f)) and
             f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Constants
NEED_COLON = [3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21]
NEXT_LINE = 9
ID_NUMBER = 3

# Initialize OCR readers
reader = easyocr.Reader(['id'])

# ------------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------------


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method in ["right-to-left", "bottom-to-top"]:
        reverse = True
    if method in ["top-to-bottom", "bottom-to-top"]:
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes


def return_id_number(image, gray):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(
        gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)
    threshCnts, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    locs = []
    for (i, c) in enumerate(threshCnts):
        (x, y, w, h) = cv2.boundingRect(c)
        if h > 10 and w > 100 and x < 300:
            locs.append((x, y, w, h, w * h))
    locs = sorted(locs, key=itemgetter(1), reverse=False)

    if len(locs) < 3:
        return ""
    nik_region = locs[1]
    nik = image[nik_region[1]-10:nik_region[1]+nik_region[3]+10,
                nik_region[0]-10:nik_region[0]+nik_region[2]+10]

    ref_img = cv2.imread(MODULE_IMAGE_PATH)
    if ref_img is None:
        raise IOError(
            f"Failed to read digit template file. Make sure '{MODULE_IMAGE_PATH}' exists and is accessible.")
    ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(ref, 66, 255, cv2.THRESH_BINARY_INV)[1]
    refCnts, hierarchy = cv2.findContours(
        ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refCnts = sort_contours(refCnts, method="left-to-right")[0]
    digits = {}
    for (i, c) in enumerate(refCnts):
        (x, y, w, h) = cv2.boundingRect(c)
        roi = ref[y:y+h, x:x+w]
        roi = cv2.resize(roi, (57, 88))
        digits[i] = roi

    gray_nik = cv2.cvtColor(nik, cv2.COLOR_BGR2GRAY)
    group = cv2.threshold(gray_nik, 127, 255, cv2.THRESH_BINARY_INV)[1]
    digitCnts, hierarchy_nik = cv2.findContours(
        group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = sort_contours(digitCnts, method="left-to-right")[0]

    groupOutput = []
    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w < 5 or h < 10:
            continue
        roi = group[y:y+h, x:x+w]
        roi = cv2.resize(roi, (57, 88))
        scores = []
        for (digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        groupOutput.append(str(np.argmax(scores)))
    return ''.join(groupOutput)


def ocr_raw(image_path):
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        raise IOError(f"Failed to read ID card image: '{image_path}'")
    image = cv2.resize(img_raw, (800, 500))
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    extracted_nik = return_id_number(image, img_gray)
    cv2.fillPoly(img_gray, pts=[np.asarray(
        [(540, 150), (540, 499), (798, 499), (798, 150)])], color=(255, 255, 255))

    # EasyOCR
    th, threshed = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
    easy_result = reader.readtext(threshed, detail=0)
    easy_result_raw = "\n".join(easy_result)

    # Pytesseract
    tess_result_raw = pytesseract.image_to_string(threshed, lang="ind")

    return easy_result_raw, tess_result_raw, extracted_nik


def strip_op(result_raw):
    result_list = result_raw.split('\n')
    return [line for line in result_list if line.strip()]


def process_result(result_list, raw_df, id_number):
    loc2index = {}
    for i, tmp_line in enumerate(result_list):
        for j, tmp_word in enumerate(tmp_line.split(' ')):
            tmp_word_clean = tmp_word.strip(':')
            scores = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word_clean, kw)
                      for kw in raw_df['keyword'].values]
            if scores and max(scores) >= 0.6:
                loc2index[(i, j)] = int(np.argmax(scores))

    last_result_list = []
    useful_info = False
    for i, tmp_line in enumerate(result_list):
        tmp_list = []
        splitted = tmp_line.split(' ')
        for j, tmp_word in enumerate(splitted):
            tmp_word_clean = tmp_word.strip(':')
            if (i, j) in loc2index:
                useful_info = True
                if raw_df['keyword'].values[loc2index[(i, j)]] == 'NEXT_LINE':
                    last_result_list.append(tmp_list)
                    tmp_list = []
                tmp_list.append(raw_df['keyword'].values[loc2index[(i, j)]])
                idx_keyword = loc2index[(i, j)]
                if idx_keyword in NEED_COLON:
                    tmp_list.append(':')
            elif tmp_word == ':' or not tmp_word.strip():
                continue
            else:
                tmp_list.append(tmp_word_clean)
        if useful_info:
            if len(last_result_list) > 2 and ':' not in tmp_list:
                last_result_list[-1].extend(tmp_list)
            else:
                last_result_list.append(tmp_list)

    for tmp_data in last_result_list:
        while '—' in tmp_data:
            tmp_data.remove('—')

        # Fuzzy matching for province & district/city
        if 'PROVINSI' in tmp_data or 'KABUPATEN' in tmp_data or 'KOTA' in tmp_data:
            candidate_province = None
            candidate_kabkot = None
            for idx, w in enumerate(tmp_data):
                w_up = w.upper()
                if w_up == 'PROVINSI' and (idx+1 < len(tmp_data)):
                    candidate_province = tmp_data[idx+1]
                if (w_up == 'KABUPATEN' or w_up == 'KOTA') and (idx+1 < len(tmp_data)):
                    candidate_kabkot = tmp_data[idx+1]
            if candidate_province is not None:
                unique_provinces = kki_prepared_df['provinsi'].unique(
                ).tolist()
                best_score_prov = 0
                best_province = candidate_province
                for p in unique_provinces:
                    score = textdistance.damerau_levenshtein.normalized_similarity(
                        candidate_province, p)
                    if score > best_score_prov:
                        best_score_prov = score
                        best_province = p
                if best_score_prov >= 0.6:
                    if candidate_province in tmp_data:
                        pos_cp = tmp_data.index(candidate_province)
                        tmp_data[pos_cp] = best_province
                if candidate_kabkot is not None:
                    subset_kabkot = kki_prepared_df[kki_prepared_df['provinsi']
                                                    == best_province]['kabupaten_kota'].unique().tolist()
                    best_score_kabkot = 0
                    best_kabkot = candidate_kabkot
                    for kk in subset_kabkot:
                        score = textdistance.damerau_levenshtein.normalized_similarity(
                            candidate_kabkot, kk)
                        if score > best_score_kabkot:
                            best_score_kabkot = score
                            best_kabkot = kk
                    if best_score_kabkot >= 0.6:
                        if candidate_kabkot in tmp_data:
                            pos_ck = tmp_data.index(candidate_kabkot)
                            tmp_data[pos_ck] = best_kabkot

        # NIK
        if 'NIK' in tmp_data:
            if len(id_number) == 16:
                while len(tmp_data) > 3:
                    tmp_data.pop()
                tmp_data[2] = id_number
            else:
                if len(tmp_data) >= 3:
                    fallback_nik = tmp_data[2]
                    fallback_nik = fallback_nik.replace(
                        "D", "0").replace("?", "7").replace("L", "1")
                    tmp_data[2] = fallback_nik

        # Religion
        if 'Agama' in tmp_data:
            for idx, w in enumerate(tmp_data):
                if w not in ['Agama', ':']:
                    scores = [textdistance.damerau_levenshtein.normalized_similarity(w, r)
                              for r in religion_df.iloc[:, 0].values]
                    if scores and max(scores) >= 0.6:
                        best_idx = int(np.argmax(scores))
                        tmp_data[idx] = religion_df.iloc[best_idx, 0]

        # Marital Status
        if 'Status' in tmp_data or 'Perkawinan' in tmp_data:
            for idx, w in enumerate(tmp_data):
                if w not in ['Status', 'Perkawinan', ':']:
                    scores = [textdistance.damerau_levenshtein.normalized_similarity(w, m)
                              for m in marriage_df.iloc[:, 0].values]
                    if scores and max(scores) >= 0.6:
                        best_idx = int(np.argmax(scores))
                        tmp_data[idx] = marriage_df.iloc[best_idx, 0]

        # Blood Type with regex fallback
        if 'Gol' in tmp_data or 'Darah' in tmp_data:
            blood_detected = False
            for idx, w in enumerate(tmp_data):
                if w not in ['Gol', 'Darah', 'Darat', ':']:
                    scores = [textdistance.damerau_levenshtein.normalized_similarity(w, b)
                              for b in blood_df.iloc[:, 0].values]
                    if scores and max(scores) >= 0.6:
                        best_idx = int(np.argmax(scores))
                        tmp_data[idx] = blood_df.iloc[best_idx, 0]
                        blood_detected = True
            if not blood_detected:
                combined_line = " ".join(tmp_data)
                pattern = r'\b(?:A|B|AB|O)[+-]?\b'
                match = re.search(pattern, combined_line, re.IGNORECASE)
                if match:
                    detected_blood = match.group(0).upper()
                    tmp_data.append(detected_blood)

        # Address
        if 'Alamat' in tmp_data:
            for i_idx in range(len(tmp_data)):
                tmp_data[i_idx] = tmp_data[i_idx].replace(
                    "!", "I").replace("1", "I").replace("i", "I")

    return last_result_list

# Function to extract structured data from results


def extract_structured_data(processed_text):
    data = {
        'NIK': None,
        'Nama': None,
        'Tempat/Tgl Lahir': None,
        'Jenis Kelamin': None,
        'Alamat': None,
        'RT/RW': None,
        'Kel/Desa': None,
        'Kecamatan': None,
        'Agama': None,
        'Status Perkawinan': None,
        'Pekerjaan': None,
        'Kewarganegaraan': None,
        'Golongan Darah': None,
        'Provinsi': None,
        'Kabupaten/Kota': None
    }

    lines = processed_text.split('\n')
    for line in lines:
        for key in data.keys():
            if key in line:
                # Extract the value after the key and any colon
                value = line.split(':', 1)[1].strip(
                ) if ':' in line else line.replace(key, '').strip()
                data[key] = value
                break

    return data

# Main execution


def main():
    start_time = time.time()

    # Prepare keyword dataframe
    raw_df = id_card_keyword_df.iloc[:, 0].to_frame()
    raw_df.columns = ['keyword']

    results = []
    structured_results = []

    # Create progress counter
    total_files = len(file_list)
    processed_files = 0

    for f in file_list:
        full_path = os.path.join(IMAGE_DIR, f)
        try:
            # Update progress
            processed_files += 1
            print(f"Processing image {processed_files}/{total_files}: {f}")

            # Run OCR with both EasyOCR and Pytesseract
            easy_result_raw, tess_result_raw, id_number = ocr_raw(full_path)

            # Process EasyOCR result
            easy_result_list = strip_op(easy_result_raw)
            easy_processed = process_result(
                easy_result_list, raw_df, id_number)

            # Process Pytesseract result
            tess_result_list = strip_op(tess_result_raw)
            tess_processed = process_result(
                tess_result_list, raw_df, id_number)

            # Combine results (prioritize EasyOCR, fallback to Pytesseract if missing)
            combined_result = {}
            for line in easy_processed:
                key = " ".join(line[:2]) if len(line) > 1 else " ".join(line)
                combined_result[key] = " ".join(line)
            for line in tess_processed:
                key = " ".join(line[:2]) if len(line) > 1 else " ".join(line)
                if key not in combined_result:
                    combined_result[key] = " ".join(line)

            final_text = "\n".join(combined_result.values())

            # Additional: Check blood type in final_text, fallback to regex if not present
            if not re.search(r'\b(?:A|B|AB|O)[+-]?\b', final_text, re.IGNORECASE):
                pattern = r'\b(?:A|B|AB|O)[+-]?\b'
                match_easy = re.search(pattern, easy_result_raw, re.IGNORECASE)
                match_tess = re.search(pattern, tess_result_raw, re.IGNORECASE)
                if match_easy:
                    detected_blood = match_easy.group(0).upper()
                    final_text += "\nGol Darah : " + detected_blood
                elif match_tess:
                    detected_blood = match_tess.group(0).upper()
                    final_text += "\nGol Darah : " + detected_blood

            results.append({"file_name": f, "processed_text": final_text})

            # Extract structured data
            structured_data = extract_structured_data(final_text)
            structured_data['file_name'] = f
            structured_results.append(structured_data)

        except Exception as e:
            print(f"Error processing {f}: {str(e)}")
            results.append(
                {"file_name": f, "processed_text": "ERROR: " + str(e)})
            structured_data = {
                key: None for key in extract_structured_data("").keys()}
            structured_data['file_name'] = f
            structured_data['Error'] = str(e)
            structured_results.append(structured_data)

    # Calculate processing time
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")

    # Save results to Excel with multiple sheets
    excel_filename = "ktp_ocr_results.xlsx"

    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Sheet 1: Raw text results
        raw_df = pd.DataFrame(results)
        raw_df.to_excel(writer, sheet_name='Raw Text Results', index=False)

        # Sheet 2: Structured data
        structured_df = pd.DataFrame(structured_results)
        structured_df.to_excel(
            writer, sheet_name='Structured Data', index=False)

        # Sheet 3: Summary statistics
        summary_data = {
            'Metric': [
                'Total ID Cards Processed',
                'Successfully Processed',
                'Failed to Process',
                'Processing Time (seconds)',
                'Average Time per Image (seconds)'
            ],
            'Value': [
                len(file_list),
                len([r for r in results if not r['processed_text'].startswith('ERROR')]),
                len([r for r in results if r['processed_text'].startswith('ERROR')]),
                f"{elapsed_time:.2f}",
                f"{elapsed_time/len(file_list):.2f}" if file_list else "N/A"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Get the workbook and sheets
        workbook = writer.book
        sheet1 = writer.sheets['Raw Text Results']
        sheet2 = writer.sheets['Structured Data']
        sheet3 = writer.sheets['Summary']

        # Format the sheets
        # Set column widths
        for sheet in [sheet1, sheet2, sheet3]:
            for col in range(1, 20):  # Adjust based on your data width
                column_letter = chr(
                    64 + col) if col < 27 else chr(64 + col // 26) + chr(64 + col % 26)
                sheet.column_dimensions[column_letter].width = 20

        # Format headers
        header_fill = PatternFill(
            start_color="DDEBF7", end_color="DDEBF7", fill_type="solid")
        header_font = Font(bold=True)
        header_alignment = Alignment(horizontal='center', vertical='center')

        for sheet in [sheet1, sheet2, sheet3]:
            for cell in sheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = header_alignment

    print(
        f"Excel file '{excel_filename}' created successfully with multiple sheets.")


if __name__ == "__main__":
    main()
