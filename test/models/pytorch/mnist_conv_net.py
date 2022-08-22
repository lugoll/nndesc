import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.onnx as onnx
from torchsummary import summary
import matplotlib.pyplot as plt


model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    nn.Conv2d(32, 64, kernel_size=(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    nn.Flatten(),
    nn.Dropout(p=0.5),
    nn.Linear(5*5*64, 10),
    nn.Softmax(dim=1)
)

if __name__ == "__main__":
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])

    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    train_set, val_set = t.utils.data.random_split(mnist_trainset, [50000, 10000])
    train_loader = t.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, )
    val_loader = t.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)

    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = t.utils.data.DataLoader(mnist_testset, batch_size=128, shuffle=True)

    net = model.to(device)

    cross_el = nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(net.parameters(), lr=0.001)  # e-1
    epoch = 0

    for epoch in range(epoch):
        net.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = net(x)
            loss = cross_el(output, y)
            loss.backward()
            optimizer.step()

        total = 0
        correct = 0

        with t.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = net(x)
                for idx, i in enumerate(output):
                    if t.argmax(i) == y[idx]:
                        correct += 1
                    total += 1

        print(f'accuracy epoch {epoch+1}: {round(correct / total, 3)}')

    total = 0
    correct = 0

    net.eval()

    with t.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = net(x)
            for idx, i in enumerate(output):
                if t.argmax(i) == y[idx]:
                    correct += 1
                total += 1

    print(f'Test accuracy epoch: {round(correct / total, 3)}')

    summary(net, (1, 28, 28))

    x = t.randn(128, 1, 28, 28, requires_grad=True, device=device)
    onnx.export(net, x, 'pytorch/saved_models/my_cnn.onnx', export_params=True, opset_version=13, training=onnx.TrainingMode.EVAL)
