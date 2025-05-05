from PIL import Image
#import ffmpeg.image
import numpy as np
import argparse
import os


def convert_bin_to_png(path, output, width, height, img_size, index):
    """
    Converte um arquivo de imagem binário para PNG.
    
    Parâmetros:
    - path: O caminho do arquivo de imagem binário.
    - output: O caminho onde o arquivo .png será salvo.
    - width: A largura da imagem em pixels.
    - height: A altura da imagem em pixels.
    - img_size: tamanho do espaco de cores da imagem.
    """
    # Lê o arquivo binário
    with open(path, 'rb') as f:
        conteudo = f.read()
    
    # Converte os dados binários para uma matriz numpy
    # Assume-se aqui uma imagem em escala de RGB (24 bits)
    imagem_array = np.frombuffer(conteudo, dtype=np.uint8).reshape((height, width, img_size))

    # Cria uma imagem usando a Pillow e salva como PNG
    image = Image.fromarray(imagem_array, mode='RGB')
    b, g, r = image.split()
    image = Image.merge("RGB", (r, g, b))
    final_output = os.path.join(output, index) + '.png' 
    image.save(final_output)
    print(f"Imagem convertida com sucesso e salva em: {final_output}")

def read_txt_file_to_list_of_lists(path_of_file):
    """
    Lê um arquivo .txt e converte cada linha em uma lista de colunas.

    Parâmetros:
    - path_of_file: O caminho para o arquivo .txt a ser lido.

    Retorna:
    - Uma lista de listas, onde cada lista interna representa as colunas de uma linha do arquivo.
    """
    lines = []
    # Abre o arquivo para leitura
    with open(path_of_file, 'r') as file:
        for line in file:
            # Divide a linha pelo delimitador para obter as colunas
            column = line.strip().split(' ')
            # Adiciona a lista de colunas à lista de linhas
            lines.append(column)

    return lines

def generate_video(diretorio, fps):
    import subprocess
    # Constrói os caminhos dos arquivos de entrada e saída...
    input_file = os.path.join(diretorio, "%d.png")
    output_file = os.path.join(diretorio, "output.mp4")

    # ...para serem usados como argumentos no comando a seguir
    cmd = "ffmpeg -y -i %s -r %d -c:v libx264 -pix_fmt yuv420p %s" %(input_file, fps, output_file)
    confirm = subprocess.run(cmd, shell= True)
    if not confirm:
        print("Falha ao construir o vídeo")
        return confirm
    else:
        return (output_file)

class ImagesFromLog():
    def __init__(self, input, output, camera, path_pos=1, width_pos=7, height_pos=8, img_size_pos=9, fps=15, create_video=0):
        # super().__init__()
        self.input        = input          
        self.output       = output         
        self.camera       = camera         
        self.path_pos     = path_pos           
        self.width_pos    = width_pos          
        self.height_pos   = height_pos         
        self.img_size_pos = img_size_pos           
        self.fps          = fps        
        self.create_video = create_video           


    def process(self):
        if not os.path.exists(self.output):
            os.makedirs(self.output)
            print("Diretorio criado com sucesso!")
        else:
            print("Diretorio ja existe!")
        
        self.input = os.path.join(self.input, self.input.split("/")[-1] + ".txt")
        data = read_txt_file_to_list_of_lists(self.input)
        for item in data:
            try:
                if(item[0] == 'CAMERA' + str(self.camera)):
                    edited_full_name = str(str(item[self.path_pos]).split('/')[-1]).replace('.image','') #.replace('.','_')
                    string_1 = edited_full_name.split('_')[0].replace('.','_')
                    string_0 = edited_full_name.split('_')[1]
                    result_path = string_0 + '_' + string_1
                    convert_bin_to_png(self.input + item[self.path_pos], self.output, int(item[self.width_pos]), int(item[self.height_pos]), int(item[self.img_size_pos]), result_path)
            except Exception as error:
                print(error)
        if self.create_video:
            video = generate_video(self.output, self.fps)
            print(f"Vídeo gerado com sucesso: {video}")

# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-i', '--input', type=str, help='Caminho absoluto para o log (exemplo: -i /dados/log_123/log.txt)')
#     parser.add_argument('-o', '--output', type=str, help='Caminho absoluto para o diretorio onde serao salvas as imagens convertidas')
#     parser.add_argument('-c', '--camera', type=str, help='Camera escolhida (1/2/..)') 
#     parser.add_argument('-p', '--path_pos', type=int, default='1', help='Posicao onde a string referente ao caminho da imagem se encontra no arquivo .txt')
#     parser.add_argument('-w', '--width_pos', type=int, default='7', help='Posicao onde o valor referente a largura da imagem se encontra no arquivo .txt')
#     parser.add_argument('-he', '--height_pos', type=int, default='8', help='Posicao onde o valor referente a altura da imagem se encontra no arquivo .txt')
#     parser.add_argument('-l', '--img_size_pos', type=int, default='9', help='Posicao onde o valor referente a tamanho do espaco de cores da imagem se encontra no arquivo .txt')
#     parser.add_argument('-v', '--create_video', type=int, default='0', help='Gerar video? 0 - nao / 1 - sim')
#     parser.add_argument('--fps', type=int, default='15', help='FPS da gravacao do log')

#     opt = parser.parse_args()
#     return opt


# def main(opt):
#     process(**vars(opt))

# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)
