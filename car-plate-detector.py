import torch
import torch.nn as nn

class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

        def forward(self, x):
            x = self.features(x)

            return x


class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, shortcut = True):
        super().__init__()

        self.shortcut = shortcut
        self.bottleneck = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, kernel_size, stride, padding),
            ConvolutionalLayer(out_channels, out_channels, kernel_size, stride, padding)
        )
        
        def forward(self, x):
            input = x
            x = self.bottleneck(x)
        
            if self.shortcut:
                x = input + x

            return x


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        output_number = in_channels // 2

        self.initial = ConvolutionalLayer(in_channels, output_number, 1, 1, 0),
        self.step1 = nn.MaxPool2d(5, 1, 2),
        self.step2 = nn.MaxPool2d(5, 1, 2),
        self.step3 = nn.MaxPool2d(5, 1, 2),
        self.step4 = ConvolutionalLayer(output_number * 4, out_channels, 1, 1, 0)


        def forward(self, x):
            initial = self.initial(x)
            x1 = self.step1(initial)
            x2 = self.step2(x1)
            x3 = self.step3(x2)

            x = torch.cat((initial, x1, x2, x3), 1)
            x = self.step4(x)

            return x



class C2f(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.initial = ConvolutionalLayer(in_channels, out_channels, 1, 1, 0)
        self.B1 = BottleNeckBlock(in_channels, out_channels, kernel_size, stride, padding, False)
        self.B2 = BottleNeckBlock(in_channels, out_channels, kernel_size, stride, padding, False)
        self.final = ConvolutionalLayer(in_channels, out_channels, 1, 1, 0)

        def forward(self, x):
            initial = self.initial(x)
            x1, x2 = torch.chunk(initial, 2, 1)
            b1 = self.B1(x2)
            b2 = self.B2(b1)

            x = torch.cat((x1, b1, b2), 1)
            final = self.final(x)
        
            return final
