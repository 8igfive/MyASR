from re import X
import numpy as np
from scipy.signal import lfilter
from tqdm import tqdm
import pdb

# def hamming(N):
#     return np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])

def segment(signal, W=256, SP=0.4, Window=None):
    if Window is None:
        Window = np.hamming(W)
    L = len(signal)
    SP = int(W * SP)
    N = int((L - W) / SP + 1)
    Index = (np.stack([np.arange(1, W + 1) for _ in range(N)], axis=0) +\
        np.concatenate([np.expand_dims(np.arange(N), axis=1) * SP for _ in range(W)], axis=1)).T
    hw = np.concatenate([np.expand_dims(Window, axis=1) for _ in range(N)], axis=1)
    return signal[Index - 1] * hw # ndarray of size=(W, N)

def vad(signal, noise, NoiseCounter=0, NoiseMargin=3, Hangover=8):
    FreqResol = len(signal)
    SpectralDist = 20 * (np.log10(signal) - np.log10(noise))    # 应当做列向量使用
    SpectralDist[SpectralDist < 0] = 0
    
    Dist = np.mean(SpectralDist)
    if Dist < NoiseMargin:
        NoiseFlag = 1
        NoiseCounter += 1
    else:
        NoiseFlag = 0
        NoiseCounter = 0
    if NoiseCounter > Hangover:
        SpeechFlag = 0
    else:
        SpeechFlag = 1
    return NoiseFlag, SpeechFlag, NoiseCounter, Dist

def OverlapAdd2(XNEW,yphase=None,windowLen=None,ShiftLen=None):
    if yphase is None:
        yphase = np.angle(XNEW)
    if windowLen is None:
        windowLen = XNEW.shape[0] * 2
    ShiftLen = int(ShiftLen)
    if ShiftLen is None:
        ShiftLen = int(windowLen / 2)
        if windowLen % 2 == 1:
            print('The shift length have to be an integer as it is the number of samples.')
            print('shift length is fixed to {}'.format(ShiftLen))

    FreqRes, FrameNum = XNEW.shape
    Spec = XNEW * np.exp(1j * yphase)

    if windowLen % 2 == 1:
        Spec = np.concatenate([Spec, np.flipud(np.conj(Spec[1 : , :]))], axis=0)
    else:
        Spec = np.concatenate([Spec, np.flipud(np.conj(Spec[1 : -1, :]))], axis=0)
    sig = np.zeros((FrameNum - 1) * ShiftLen + windowLen) # 应当做列向量使用
    weight = np.zeros((FrameNum-1) * ShiftLen + windowLen) # 应当做列向量使用
    for i in range(1, FrameNum + 1):
        start = (i - 1) * ShiftLen + 1
        spec = Spec[:, i - 1]
        sig[start - 1 : start + windowLen - 1] += np.fft.ifft(spec, n=windowLen, axis=0).real
    return sig

def wienerScalart(signal, fs, IS=0.25): # signal 应当作列向量使用
    W = int(0.025 * fs)
    SP = 0.4
    wnd = np.hamming(W)

    pre_emph = 0
    signal = lfilter([1, -pre_emph], [1], signal)

    NIS = int((IS * fs -W) / (SP * W) + 1)
    
    y = segment(signal, W, SP, wnd)
    Y = np.fft.fft(y, axis=0)
    YPhase = np.angle(Y[0: int(Y.shape[0]/2) + 1, :])
    Y = np.abs(Y[0: int(Y.shape[0]/2) + 1, :])
    numberOfFrames = Y.shape[1]
    FreqResol = Y.shape[0]

    N = np.mean(Y[:, 0: NIS], axis=1) # 在之后应该以列向量的形式使用
    LambdaD = np.mean(Y[:, 0: NIS] ** 2, axis=1) # 在之后应该以列向量的形式使用
    if 0 in LambdaD:
        return signal
    alpha = 0.99
    NoiseCounter = 0
    NoiseLength = 9
    G = np.ones_like(N) # 应被用作列向量
    Gamma = np.ones_like(N) # 应被用作列向量
    X = np.zeros_like(Y)

    for i in range(1, numberOfFrames + 1):
        if i <= NIS:
            SpeechFlag = 0
            NoiseCounter = 100
        else:
            NoiseFlag, SpeechFlag, NoiseCounter, Dist = vad(Y[:, i - 1], N, NoiseCounter)

        if SpeechFlag == 0:
            N = (NoiseLength * N + Y[:, i - 1]) / (NoiseLength + 1)
            LambdaD = (NoiseLength * LambdaD + Y[:, i - 1] ** 2) / (NoiseLength + 1)
        
        gammaNew = Y[:, i - 1] ** 2 / LambdaD
        _gammaNew = gammaNew - 1
        _gammaNew[_gammaNew < 0] = 0
        xi = alpha * (G ** 2) * Gamma + (1 - alpha) * _gammaNew
        Gamma = gammaNew


        G = xi / (xi + 1)
        X[:, i - 1] = G * Y[:, i - 1]
    output = OverlapAdd2(X, YPhase, W, SP * W)
    output = lfilter([1], [1, -pre_emph], output)
    return output