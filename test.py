# -*- coding: utf-8 -*-
"""
docparser 테스트 스크립트

CLI 로직을 직접 재사용하여 PDF 문서를 파싱합니다.
"""

import sys
from docparser.cli import run_parse

def main():
    """
    docparser CLI의 핵심 로직을 직접 호출하여 테스트합니다.
    사용된 명령어:
    uv run docparser-parse \
        --pdf temp.pdf \
        --out_dir ./output/test \
        --detector doclayout-yolo \
        --yolo-weights "juliozhao/DocLayout-YOLO-DocStructBench" \
        --yolo-from-pretrained \
        --ocr-provider deepseek-ollama \
        --ocr-scope elements \
        --page-end 1 \
        --vlm-provider vlm-openai \
        --vlm-scope visuals \
        --vlm-use-image \
        --execution-mode parallel \
        --md-style final
    """
    
    print("[TEST] test.py 실행 시작...")
    
    try:
        run_parse(
            pdf="temp.pdf",
            out_dir="./output/test",
            detector="doclayout-yolo",
            yolo_weights="juliozhao/DocLayout-YOLO-DocStructBench",
            yolo_from_pretrained=True,
            ocr_provider="glm-ocr",
            ocr_scope="page",
            page_end=1,
            vlm_provider="vlm-openai",
            vlm_scope="visuals",
            vlm_use_image=True,
            execution_mode="parallel",
            parallel_workers=1,
            md_style="final",
            save_detections=True,
        )
        print("[TEST] 파싱이 성공적으로 완료되었습니다.")
        
    except Exception as e:
        print(f"[TEST] 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()