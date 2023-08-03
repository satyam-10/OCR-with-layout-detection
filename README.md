# OCR-with-layout-detection
The project "OCR with Layout Detection" aims to automatically extract text from images or scanned documents while preserving the layout structure, such as paragraphs, headings, tables, and figures. The combination of OCR (Optical Character Recognition) and layout detection enables efficient information extraction from documents with complex structures, making it easier to process and analyze large volumes of textual data.
OCR was done using Tesseract
### Tesseract OCR
Tesseract is an open source text recognition (OCR) Engine, available under the Apache 2.0 license. It can be used directly, or (for programmers) using an API to extract printed text from images. It supports a wide variety of languages. Tesseract doesn't have a built-in GUI, but there are several available from the 3rdParty page. Tesseract is compatible with many programming languages and frameworks through wrappers that can be found here. It can be used with the existing layout analysis to recognize text within a large document, or it can be used in conjunction with an external text detector to recognize text from an image of a single text line.
### Installing Tesseract
download the tesseract.exe and copy the path where it is installed.
```ruby
import pytesseract 
pytesseract.pytesseract.tesseract_cmd="C:/Program Files/Tesseract-OCR/tesseract.exe"
```
### layout parser
A Unified Toolkit for Deep Learning Based Document Image Analysis.
installation:-
```ruby
import layoutparser as lp
model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.65],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
layout = model.detect(image) 
lp.draw_box(image, layout,)
```
