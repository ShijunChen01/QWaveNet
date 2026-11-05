import pennylane as qml
import torch
import torch.nn as nn

from layers.Embed import PatchEmbedding
from layers.EncDec import Decoder
from layers.QuantumWaveletFusion import fuse_quantum_wavelet
from layers.WaveletEmbedding import WaveletEmbedding
from layers.QuantumMeasurement import quantum_measurement


class QuantumConv1D(nn.Module):

    def __init__(self, d_model=128, dec_in=321, compressed_time=4, batch=32):
        super().__init__()

        self.n_qubits = compressed_time
        self.qbatch = 2 * compressed_time
        self.weight = nn.Parameter(torch.randn(self.n_qubits))

        self.compressor = nn.Linear(batch, self.qbatch)
        self.temporal_compressor = nn.Linear(d_model, self.n_qubits)
        self.temporal_compressor1 = nn.Linear(dec_in, self.n_qubits)

        self.restorer = nn.Linear(self.qbatch, batch)
        self.temporal_restorer = nn.Linear(self.n_qubits, d_model)
        self.temporal_restorer1 = nn.Linear(self.n_qubits, dec_in)

        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.qnode = qml.QNode(self.quantum_conv1d, self.dev, interface="torch")

    def quantum_conv1d(self, inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation="Y")

        for i in range(self.n_qubits - 1):
            qml.RY(weights[i], wires=i + 1)
            qml.CNOT(wires=[i, i + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.temporal_compressor(x)
        x = x.permute(0, 2, 1).permute(2, 1, 0)
        x = self.compressor(x).permute(2, 1, 0)

        _, time, feat = x.shape
        x = self.temporal_compressor1(x.reshape(-1, feat))

        outputs = []
        for inputs in x:
            outputs.append(quantum_measurement(self.qnode, inputs, self.weight))

        x = torch.stack(outputs).to(x.device)
        x = x.reshape(self.qbatch, time, x.shape[-1])
        x = x.permute(2, 1, 0)
        x = self.restorer(x).permute(2, 1, 0)
        x = self.temporal_restorer1(x).permute(0, 2, 1)
        x = self.temporal_restorer(x).permute(0, 2, 1)

        return x




class Model(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.use_norm = args.use_norm
        self.pred_len = args.pred_len
        self.device = torch.device(f"cuda:{args.gpu}")

        self.enc_embedding = PatchEmbedding(
            input_len=args.seq_len,
            d_model=args.d_model,
        )

        # QFE
        self.qtrend = QuantumConv1D(
            d_model=args.d_model,
            dec_in=args.dec_in,
            compressed_time=args.compressed_time,
            batch=args.batch_size,
        )

        # GAWT
        self.swt = WaveletEmbedding(
            d_channel=args.dec_in,
            d_model=args.d_model,
            swt=True,
            requires_grad=args.requires_grad,
            wv=args.wv,
            kernel_size=args.kernel_size,
        )

        # ODP
        self.decoder = Decoder(
            d_model=args.d_model,
            num_layers=args.d_layers,
            num_heads=args.n_heads,
            feedforward_dim=args.d_ff,
            dropout=args.dropout,
            mask_ratio=args.mask_ratio,
        )
        self.param1 = nn.Parameter(torch.tensor([0.5]))

        self.out_projection = nn.Sequential(
            nn.Linear(args.d_model, args.d_model),
            WaveletEmbedding(
                d_channel=args.dec_in,
                d_model=args.d_model,
                swt=False,
                requires_grad=args.requires_grad,
                wv=args.wv,
                kernel_size=args.kernel_size,
            ),
        )
        self.forecast_head = nn.Linear(args.d_model, args.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdevs = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdevs

        x = self.enc_embedding(x_enc.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        # QFE: dynamic evolution and evolutionary residual
        DE = self.qtrend(x).to(self.device)
        ER = x - DE

        # GAWT + ODP: quantum-wavelet integration
        coeffs = fuse_quantum_wavelet(x, ER, DE, self.swt, self.decoder, self.param1)

        output = self.out_projection(coeffs)
        output = self.forecast_head(output).permute(0, 2, 1)

        if self.use_norm:
            output = output * stdevs[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )
            output = output + means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )

        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, None, None, None)
