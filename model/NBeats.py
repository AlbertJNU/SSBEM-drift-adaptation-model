from functools import partial
from typing import List, Tuple, Union, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import rnn


def to_list(value: Any) -> List[Any]:
    if isinstance(value, (tuple, list)) and not isinstance(value, rnn.PackedSequence):
        return value
    else:
        return [value]

class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, interpolation_mode:str):
        super(IdentityBasis, self).__init__()
        assert (interpolation_mode in ['linear', 'nearest']) or ('cubic' in interpolation_mode)
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode

    def forward(
            self,
            theta
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        backcast = theta[:, : self.backcast_size]
        knots = theta[:, self.backcast_size:]

        if self.interpolation_mode == "nearest":
            knots = knots[:, None, :]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode)
            forecast = forecast[:, 0, :]
        elif self.interpolation_mode == "linear":
            knots = knots[:, None, :]
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode
            )  # , align_corners=True)
            forecast = forecast[:, 0, :]
        elif "cubic" in self.interpolation_mode:
            batch_size = int(self.interpolation_mode.split("-")[-1])
            knots = knots[:, None, None, :]
            forecast = torch.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size: (i + 1) * batch_size], size=self.forecast_size, mode="bicubic"
                )  # , align_corners=True)
                forecast[i * batch_size: (i + 1) * batch_size] += forecast_i[:, 0, 0, :]

        return backcast, forecast

class TrendBasis(nn.Module):
    def __init__(
        self,
        degree_of_polynomial: int,
        backcast_size: int,
        forecast_size: int,
        interpolation_mode: str,
        out_features: int = 1,
    ):
        super().__init__()
        assert (interpolation_mode in ['linear', 'nearest']) or ('cubic' in interpolation_mode)
        self.out_features = out_features
        polynomial_size = degree_of_polynomial + 1
        self.backcast_basis = nn.Parameter(
            torch.tensor(
                np.concatenate(
                    [
                        np.power(
                            np.arange(backcast_size, dtype=float) / backcast_size, i
                        )[None, :]
                        for i in range(polynomial_size)
                    ]
                ),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.forecast_basis = nn.Parameter(
            torch.tensor(
                np.concatenate(
                    [
                        np.power(
                            np.arange(forecast_size, dtype=float) / forecast_size, i
                        )[None, :]
                        for i in range(polynomial_size)
                    ]
                ),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.interpolation_mode = interpolation_mode
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size


    def forward(self,
            theta,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        polynomial_size = self.forecast_basis.shape[0]
        backcast_theta = theta[:, :polynomial_size]
        forecast_theta = theta[:, :polynomial_size]
        backcast = torch.matmul(backcast_theta, self.backcast_basis)
        forecast = torch.matmul(forecast_theta, self.forecast_basis)

        backcast = backcast
        knots = forecast

        if self.interpolation_mode == "nearest":
            knots = knots[:, None, :]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode)
            forecast = forecast[:, 0, :]
        elif self.interpolation_mode == "linear":
            knots = knots[:, None, :]
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode
            )
            forecast = forecast[:, 0, :]
        elif "cubic" in self.interpolation_mode:
            batch_size = int(self.interpolation_mode.split("-")[-1])
            knots = knots[:, None, None, :]
            forecast = torch.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size: (i + 1) * batch_size], size=self.forecast_size, mode="bicubic"
                )
                forecast[i * batch_size: (i + 1) * batch_size] += forecast_i[:, 0, 0, :]

        return backcast, forecast


class SeasonalityBasis(nn.Module):
    def __init__(
        self,
        harmonics: int,
        backcast_size: int,
        forecast_size: int,
        interpolation_mode: str,
        out_features: int = 1,
    ):
        super().__init__()
        assert (interpolation_mode in ['linear', 'nearest']) or ('cubic' in interpolation_mode)
        self.out_features = out_features
        frequency = np.append(
            np.zeros(1, dtype=float),
            np.arange(harmonics, harmonics / 2 * forecast_size, dtype=float)
            / harmonics,
        )[None, :]
        backcast_grid = (
            -2
            * np.pi
            * (np.arange(backcast_size, dtype=float)[:, None] / forecast_size)
            * frequency
        )
        forecast_grid = (
            2
            * np.pi
            * (np.arange(forecast_size, dtype=float)[:, None] / forecast_size)
            * frequency
        )

        backcast_cos_template = torch.tensor(
            np.transpose(np.cos(backcast_grid)), dtype=torch.float32
        )
        backcast_sin_template = torch.tensor(
            np.transpose(np.sin(backcast_grid)), dtype=torch.float32
        )
        backcast_template = torch.cat(
            [backcast_cos_template, backcast_sin_template], dim=0
        )

        forecast_cos_template = torch.tensor(
            np.transpose(np.cos(forecast_grid)), dtype=torch.float32
        )
        forecast_sin_template = torch.tensor(
            np.transpose(np.sin(forecast_grid)), dtype=torch.float32
        )
        forecast_template = torch.cat(
            [forecast_cos_template, forecast_sin_template], dim=0
        )
        self.interpolation_mode = interpolation_mode
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)

    def forward(self,
            theta,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        harmonic_size = self.forecast_basis.shape[0]
        backcast_theta = theta[:, :harmonic_size]
        forecast_theta = theta[:, :harmonic_size]
        backcast = torch.matmul(backcast_theta, self.backcast_basis)
        forecast = torch.matmul(forecast_theta, self.forecast_basis)

        backcast = backcast
        knots = forecast

        if self.interpolation_mode == "nearest":
            knots = knots[:, None, :]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode)
            forecast = forecast[:, 0, :]
        elif self.interpolation_mode == "linear":
            knots = knots[:, None, :]
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode
            )
            forecast = forecast[:, 0, :]
        elif "cubic" in self.interpolation_mode:
            batch_size = int(self.interpolation_mode.split("-")[-1])
            knots = knots[:, None, None, :]
            forecast = torch.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size: (i + 1) * batch_size], size=self.forecast_size, mode="bicubic"
                )
                forecast[i * batch_size: (i + 1) * batch_size] += forecast_i[:, 0, 0, :]

        return backcast, forecast


def init_weights(module, initialization):
    if type(module) == torch.nn.Linear:
        if initialization == "orthogonal":
            torch.nn.init.orthogonal_(module.weight)
        elif initialization == "he_uniform":
            torch.nn.init.kaiming_uniform_(module.weight)
        elif initialization == "he_normal":
            torch.nn.init.kaiming_normal_(module.weight)
        elif initialization == "glorot_uniform":
            torch.nn.init.xavier_uniform_(module.weight)
        elif initialization == "glorot_normal":
            torch.nn.init.xavier_normal_(module.weight)
        elif initialization == "lecun_normal":
            pass
        else:
            assert 1 < 0, f"Initialization {initialization} not found"

ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]


class NBEATSBlock(nn.Module):

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        output_size: Union[int, List[int]],
        covariate_size: int,
        n_theta: int,
        hidden_size: List[int],
        pooling_sizes: int,
        pooling_mode: str,
        basis: nn.Module,
        n_layers: int,
        batch_normalization: bool,
        dropout: float,
        activation: str,
    ):
        super().__init__()

        assert pooling_mode in ["max", "average"]

        self.context_length_pooled = int(np.ceil(context_length / pooling_sizes))

        self.context_length = context_length
        self.output_size = output_size
        self.n_theta = n_theta
        self.prediction_length = prediction_length
        self.covariate_size = covariate_size
        self.pooling_sizes = pooling_sizes
        self.batch_normalization = batch_normalization
        self.dropout = dropout

        self.hidden_size = [
            self.context_length_pooled * len(self.output_size)
            + (self.context_length + self.prediction_length) * self.covariate_size
        ] + hidden_size

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        activ = getattr(nn, activation)()

        if pooling_mode == "max":
            self.pooling_layer = nn.MaxPool1d(kernel_size=self.pooling_sizes, stride=self.pooling_sizes, ceil_mode=True)
        elif pooling_mode == "average":
            self.pooling_layer = nn.AvgPool1d(kernel_size=self.pooling_sizes, stride=self.pooling_sizes, ceil_mode=True)

        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(nn.Linear(in_features=self.hidden_size[i], out_features=self.hidden_size[i + 1]))
            hidden_layers.append(activ)

            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=self.hidden_size[i + 1]))

            if self.dropout > 0:
                hidden_layers.append(nn.Dropout(p=self.dropout))

        output_layer = [
            nn.Linear(
                in_features=self.hidden_size[-1],
                out_features=n_theta,
            )
        ]
        layers = hidden_layers + output_layer

        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(
        self, encoder_y: torch.Tensor, encoder_x_t: torch.Tensor, decoder_x_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(encoder_y)

        encoder_y = encoder_y.transpose(1, 2)
        encoder_y = self.pooling_layer(encoder_y)
        encoder_y = encoder_y.transpose(1, 2).reshape(batch_size, -1)

        if self.covariate_size > 0:
            encoder_y = torch.cat(
                (
                    encoder_y,
                    encoder_x_t.reshape(batch_size, -1),
                    decoder_x_t.reshape(batch_size, -1),
                ),
                1,
            )

        theta = self.layers(encoder_y)
        backcast, forecast = self.basis(theta)
        backcast = backcast.reshape(-1, len(self.output_size), self.context_length).transpose(1, 2)
        forecast = forecast.reshape(-1, sum(self.output_size), self.prediction_length).transpose(1, 2)

        return backcast, forecast


class NBEATSModule(nn.Module):

    def __init__(
        self,
        h,
        context_length,
        prediction_length,
        n_polynomials: int,
        n_harmonics: int,
        output_size: int,
        covariate_size,
        n_blocks: list,
        n_layers: list,
        hidden_size: list,
        pooling_sizes: list,
        downsample_frequencies: list,
        stack_types: list,
        pooling_mode,
        interpolation_mode,
        dropout,
        activation,
        initialization,
        batch_normalization,
        shared_weights,
        naive_level: bool,
    ):
        super().__init__()

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.output_size = output_size
        self.naive_level = naive_level

        blocks = self.create_stack(
            h=h,
            n_blocks=n_blocks,
            context_length=context_length,
            prediction_length=prediction_length,
            output_size=output_size,
            covariate_size=covariate_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            pooling_sizes=pooling_sizes,
            stack_types=stack_types,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            batch_normalization=batch_normalization,
            dropout=dropout,
            activation=activation,
            shared_weights=shared_weights,
            initialization=initialization,
            n_polynomials=n_polynomials,
            n_harmonics=n_harmonics,
        )
        self.blocks = torch.nn.ModuleList(blocks)

    def create_stack(
        self,
        h,
        n_blocks,
        context_length,
        prediction_length,
        output_size,
        covariate_size,
        n_layers,
        hidden_size,
        pooling_sizes,
        stack_types,
        pooling_mode,
        interpolation_mode,
        batch_normalization,
        dropout,
        activation,
        shared_weights,
        initialization,
        n_polynomials,
        n_harmonics,
    ):

        block_list = []
        for i in range(len(n_blocks)):
            for block_id in range(n_blocks[i]):

                if (len(block_list) == 0) and (batch_normalization):
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False

                if shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:
                    if stack_types[i] == "identity":
                        n_theta = context_length + prediction_length
                        basis = IdentityBasis(
                            backcast_size=context_length,
                            forecast_size=prediction_length,
                            interpolation_mode=interpolation_mode,
                        )
                    elif stack_types[i] == "seasonality":
                        n_theta = 2*int(np.ceil(n_harmonics / 2 * h) - (n_harmonics - 1))
                        basis = SeasonalityBasis(
                            harmonics=n_harmonics,
                            backcast_size=context_length,
                            forecast_size=prediction_length,
                            interpolation_mode=interpolation_mode,
                            out_features=1,
                        )
                    elif stack_types[i] == "trend":
                        n_theta = (
                            n_polynomials + 1
                        )
                        basis = TrendBasis(
                            degree_of_polynomial=n_polynomials,
                            backcast_size=context_length,
                            forecast_size=prediction_length,
                            interpolation_mode=interpolation_mode,
                            out_features=1,
                        )
                    else:
                        raise ValueError(f"Block type {stack_types[i]} not found!")

                    nbeats_block = NBEATSBlock(
                        context_length=context_length,
                        prediction_length=prediction_length,
                        output_size=output_size,
                        covariate_size=covariate_size,
                        n_theta=n_theta,
                        hidden_size=hidden_size[i],
                        pooling_sizes=pooling_sizes[i],
                        pooling_mode=pooling_mode,
                        basis=basis,
                        n_layers=n_layers[i],
                        batch_normalization=batch_normalization_block,
                        dropout=dropout,
                        activation=activation,
                    )

                init_function = partial(init_weights, initialization=initialization)
                nbeats_block.layers.apply(init_function)
                block_list.append(nbeats_block)
        return block_list

    def forward(
        self,
        encoder_y,
        encoder_x_t,
        decoder_x_t,
    ):

        residuals = (
            encoder_y
        )
        level = encoder_y[:, -1:].repeat(1, self.prediction_length, 1)  # Level with Naive1
        forecast_level = level.repeat_interleave(torch.tensor(self.output_size, device=level.device), dim=2)

        if self.naive_level:
            block_forecasts = [forecast_level]
            block_backcasts = [encoder_y[:, -1:].repeat(1, self.context_length, 1)]

            forecast = block_forecasts[0]
        else:
            block_forecasts = []
            block_backcasts = []
            forecast = torch.zeros_like(forecast_level, device=forecast_level.device)

        for block in self.blocks:
            block_backcast, block_forecast = block(
                encoder_y=residuals, encoder_x_t=encoder_x_t, decoder_x_t=decoder_x_t
            )

            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)
            block_backcasts.append(block_backcast)

        block_forecasts = torch.stack(block_forecasts, dim=-1)
        block_backcasts = torch.stack(block_backcasts, dim=-1)
        backcast = residuals

        return forecast, backcast, block_forecasts, block_backcasts


class NBEATS(nn.Module):
    def __init__(
        self,
        h=2,
        output_size: Union[int, List[int]] = 1,
        n_harmonics: int = 2,
        n_polynomials: int = 2,
        context_length: int = 1,
        covariate_size: int = 1,
        prediction_length: int = 1,
        naive_level: bool = True,
        shared_weights: bool = True,
        activation: str = "ReLU",
        initialization: str = "lecun_normal",
        n_blocks: List[int] = [1, 1, 1],
        n_layers: Union[int, List[int]] = 6,
        hidden_size: int = 512,
        pooling_sizes: Optional[List[int]] = None,
        downsample_frequencies: Optional[List[int]] = None,
        stack_types: list = ['identity', 'trend', 'seasonality'],
        pooling_mode: str = "max",
        interpolation_mode: str = "linear",
        batch_normalization: bool = False,
        dropout: float = 0.0,
    ):

        self.output_size = output_size
        self.covariate_size = covariate_size


        if activation == "SELU":
            initialization = "lecun_normal"

        n_stacks = len(n_blocks)
        if pooling_sizes is None:
            pooling_sizes = np.exp2(np.round(np.linspace(0.49, np.log2(prediction_length / 2), n_stacks)))
            pooling_sizes = [int(x) for x in pooling_sizes[::-1]]
        if downsample_frequencies is None:
            downsample_frequencies = [min(prediction_length, int(np.power(x, 1.5))) for x in pooling_sizes]

        if isinstance(n_layers, int):
            n_layers = [n_layers] * n_stacks

        super(NBEATS, self).__init__()

        self.model = NBEATSModule(
            h=h,
            output_size=to_list(output_size),
            context_length=context_length,
            prediction_length=prediction_length,
            n_harmonics=n_harmonics,
            n_polynomials=n_polynomials,
            covariate_size=covariate_size,
            n_blocks=n_blocks,
            n_layers=n_layers,
            hidden_size=n_stacks * [6 * [hidden_size]],
            pooling_sizes=pooling_sizes,
            downsample_frequencies=downsample_frequencies,
            stack_types=stack_types,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            dropout=dropout,
            activation=activation,
            initialization=initialization,
            batch_normalization=batch_normalization,
            shared_weights=shared_weights,
            naive_level=naive_level,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[Any, Any, Tuple[Any, ...], Tuple[Any, ...]]:

        if self.covariate_size > 0:
            encoder_x_t = x[:, :-1, :]
            decoder_x_t = x[:, -1, :].unsqueeze(1)
        else:
            encoder_x_t = None
            decoder_x_t = None

        encoder_y = y

        forecast, backcast, block_forecasts, block_backcasts = self.model(
            encoder_y, encoder_x_t, decoder_x_t
        )
        backcast = encoder_y - backcast

        block_backcasts = block_backcasts.detach()
        block_forecasts = block_forecasts.detach()

        if isinstance(self.output_size, (tuple, list)):
            forecast = forecast.split(self.output_size, dim=2)
            backcast = backcast.split(1, dim=2)
            block_backcasts = tuple(
                (block.squeeze(3).split(1, dim=2)) for block in block_backcasts.split(1, dim=3)
            )
            block_forecasts = tuple(
                (block.squeeze(3).split(self.output_size, dim=2)) for block in block_forecasts.split(1, dim=3)
            )

        return forecast, backcast, block_backcasts, block_forecasts

