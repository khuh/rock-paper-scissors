import cv2
import os
import glob

이미지를 반전시킬 경로 설정 (입력 경로와 출력 경로를 동일하게 설정)
directory = './dataset/valid/paper'  # 원본 이미지들이 있는 경로

지원할 이미지 확장자들 (필요시 확장자 추가 가능)
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '*.tiff']

각 확장자별로 파일 리스트 가져오기
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(directory, ext)))

이미지 파일들 처리
for image_path in image_files:
    # 파일명과 확장자 분리
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)

    # 파일명이 '_flipped'로 끝나면 건너뛰기
    if name.endswith('_flipped'):
        print(f"이미 반전된 파일입니다: {image_path}")
        continue

    # 이미지 읽기
    image = cv2.imread(image_path)

    # 이미지가 제대로 읽혔는지 확인
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        continue

    # 이미지 좌우 반전
    flipped_image = cv2.flip(image, 1)

    # 반전된 이미지 저장 경로 생성 (입력 경로와 동일한 위치에 저장)
    flipped_filename = f"{name}_flipped{ext}"
    flipped_image_path = os.path.join(directory, flipped_filename)

    # 반전된 이미지 저장
    cv2.imwrite(flipped_image_path, flipped_image)

    print(f"반전된 이미지가 저장되었습니다: {flipped_image_path}") 
