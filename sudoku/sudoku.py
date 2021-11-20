import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border

sudoku = 'sudoku.png'
plt.style.use("dark_background")
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
num_list = ['0','1','2','3','4','5','6','7','8','9']


def possible(row,column,board,value):
    for x  in range(0,9):
        if board[row][x] == value or board[x][column] == value:
            return False

    row = (row//3)*3
    column = (column//3)*3
    for i in range(row,row+3):
        for j in range(column,column+3):
            if board[i][j] == value:
                return False
    return True

def solve(board):
    for x in range(9):
        for y in range(9):
            if board[x][y] == 0:
                for z in range(1,10):
                    if possible(x,y,board,z):
                        board[x][y] = z
                        solve(board)
                        if len(np.where(np.array(board) == 0)[0]) == 0:return np.array(board)
                        board[x][y] = 0
                return

def run(image):
    img = cv2.imread(image)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 1)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
    mask = np.zeros_like(rgb)
    output = np.zeros_like(rgb)

    max_area = 0
    ma_index = 0
    contours, hir = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for index,contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            ma_index = index
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    puzzle = four_point_transform(rgb, approx.reshape(4, 2))
    warped = four_point_transform(gray, approx.reshape(4, 2))

    empty_cell = []
    # build the sudoku board
    board = np.arange(81).reshape(9,9)
    # cell for the sudoku
    cell_height = warped.shape[0] // 9
    cell_width = warped.shape[1] // 9

    celly = 0
    for y in range(9):
        cellx = 0
        for x in range(9):
            cell = warped[celly:celly+cell_height, cellx:cellx+cell_width]
            cell = cv2.resize(cell, (128,128)) # resize to high scale to remove checkerboard effect
            _,binary = cv2.threshold(cell, 80, 255, cv2.THRESH_BINARY_INV)
            binary = clear_border(binary)
            digit = pytesseract.image_to_string(binary, config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')[0]
            if digit not in num_list:
                digit = 0
                empty_cell.append([cellx,celly,x,y])
            board[y,x] = digit
            cellx += cell_width
        celly += cell_height

    solved_board = solve(board)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for cell in empty_cell:
        cv2.putText(puzzle, str(solved_board[cell[-1]][cell[-2]]), (cell[0]+25,cell[1]+45), font, 1, (255,0,0), 2)

    rgb = cv2.resize(rgb, (puzzle.shape[1],puzzle.shape[0]))
    combined = cv2.hconcat((rgb,puzzle))
    plt.title("Left - original, Right - solved")
    plt.imshow(combined, cmap = 'gray')
    plt.show()

if __name__ == '__main__':
    run(sudoku)
