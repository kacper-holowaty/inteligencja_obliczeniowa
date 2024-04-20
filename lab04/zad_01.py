# SIATKÓWKA NA PLAŻY

# wiek waga wzrost gra
# 23 75 176 True
# 25 67 180 True
# 28 120 175 False
# 22 65 165 True
# 46 70 187 True
# 50 68 180 False
# 48 97 178 False


import math

def forwardPass(wiek, waga, wzrost): 
    w1_wiek, w1_waga, w1_wzrost = -0.46122, 0.97314, -0.39203
    w2_wiek, w2_waga, w2_wzrost = 0.78548, 2.10584, -0.57847
    
    bias_hidden1, bias_hidden2 = 0.80109, 0.43529
    
    hidden1 = wiek * w1_wiek + waga * w1_waga + wzrost * w1_wzrost + bias_hidden1
    hidden1_po_aktywacji = 1 / (1 + math.exp(-hidden1))
    
    hidden2 = wiek * w2_wiek + waga * w2_waga + wzrost * w2_wzrost + bias_hidden2
    hidden2_po_aktywacji = 1 / (1 + math.exp(-hidden2))
    
    w_output_hidden1, w_output_hidden2, bias_output = -0.81546, 1.03773, -0.2368
    
    output = hidden1_po_aktywacji * w_output_hidden1 + hidden2_po_aktywacji * w_output_hidden2 + bias_output
    
    return output  

print(forwardPass(25, 67, 180))