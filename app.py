import cv2
import streamlit as st
from pdf2image import convert_from_bytes
import layoutparser as lp
import numpy as np

def main():
    st.title("OCR with Layout Detection")

    # File upload
    uploaded_file = st.file_uploader("Upload a PNG/PDF file", type=["png", "pdf"])

    if uploaded_file is not None:
        # Read the uploaded image or PDF
        if uploaded_file.type == "image/png":
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        else:  
            images = convert_from_bytes(uploaded_file.read())
            if images:
                image = np.array(images[0])
            else:
                image = None
        if image is not None:
            image = image[..., ::-1] 
            model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.65],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
            layout = model.detect(image)
            figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])
            text_blocks = lp.Layout([b for b in text_blocks \
                   if not any(b.is_in(b_fig) for b_fig in figure_blocks)])

            # OCR with TesseractAgent
            ocr_agent = lp.TesseractAgent(languages='eng')
            for block in text_blocks:
                segment_image = (block
                                 .pad(left=5, right=5, top=5, bottom=5)
                                 .crop_image(image))
                text = ocr_agent.detect(segment_image)
                block.set(text=text, inplace=True)

            # Display the detected text
            st.subheader("Detected Text:")
            for block in text_blocks:
                if block.text:
                    st.write(block.text)
                    st.write("---")

            # Display the image with bounding boxes (optional)
            st.subheader("Layout Detection:")
            annotated_image = lp.draw_box(image, layout, box_width=3, show_element_id=True)
            st.image(annotated_image, use_column_width=True)

if __name__ == "__main__":
    main()


