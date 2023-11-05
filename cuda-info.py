import torch

def main():
    avb = torch.cuda.is_available()
    print(f"Cuda available: {avb}")
    if avb:
        num_cuda = torch.cuda.device_count()
        print(f"Num cuda: {avb}")
        for i in range(num_cuda):
            print(f"cuda{i}: {torch.cuda.get_device_name(i)}")
        

if __name__ == "__main__":
    main()