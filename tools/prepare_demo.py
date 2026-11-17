"""Generate an artificial input; contains no laboratory images or annotations."""
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    output=Path(__file__).resolve().parents[1]/'examples'
    output.mkdir(exist_ok=True)
    image=Image.new('RGB',(641,385),(128,128,128))
    draw=ImageDraw.Draw(image)
    for box in [(20,15,160,160),(210,0,360,120),(430,30,605,185),(0,235,145,380),(245,215,410,380),(495,270,665,410)]:
        draw.ellipse(box,fill='white')
    for box in [(162,127,206,201),(375,135,432,224),(155,274,214,312),(441,205,505,253),(350,145,365,175)]:
        draw.ellipse(box,fill='red')
    image.save(output/'synthetic_input.png')
    print('Generated synthetic example (not experimental data)')


if __name__=='__main__':
    main()
