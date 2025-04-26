from functools import partial
from typing import List, Tuple, Union, Optional, Any
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
            backcast_theta: torch.Tensor,
            forecast_theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        backcast = backcast_theta
        knots = forecast_theta

        if self.interpolation_mode == "nearest":
            knots = knots[:, None, :]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode, align_corners=True)
            forecast = forecast[:, 0, :]
        elif self.interpolation_mode == "linear":
            knots = knots[:, None, :]
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode, align_corners=True
            )
            forecast = forecast[:, 0, :]
        elif "cubic" in self.interpolation_mode:
            batch_size = int(self.interpolation_mode.split("-")[-1])
            knots = knots[:, None, None, :]
            forecast = torch.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size: (i + 1) * batch_size], size=self.forecast_size, mode="bicubic", align_corners=True
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

class NEATSBlock(nn.Module):

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
        n_layers: int,
        batch_normalization: bool,
        dropout: float,
        activation: str,
        use_predict_covariate: bool,
        n_delays: int,
        svd_low_rank: int
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
        self.use_predict_covariate = use_predict_covariate
        self.n_delays = n_delays
        self.svd_low_rank = svd_low_rank

        if self.use_predict_covariate == True:
            self.hidden_size = [
                                   self.context_length_pooled * len(self.output_size)
                                   + (self.context_length + self.prediction_length) * self.covariate_size
                               ] + hidden_size
        else:
            self.hidden_size = [
                                   self.context_length_pooled * len(self.output_size)
                                   + (self.context_length) * self.covariate_size
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
                out_features=self.svd_low_rank,
            )
        ]
        layers = hidden_layers + output_layer

        self.layers_backcast = nn.Sequential(*layers)
        self.layers_forecast = nn.Sequential(*layers)

    def forward(
        self, encoder_y: torch.Tensor, encoder_x_t: torch.Tensor, decoder_x_t: torch.Tensor, specific_basis,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = len(encoder_y)

        encoder_y = encoder_y.transpose(1, 2)
        encoder_y = self.pooling_layer(encoder_y)
        encoder_y = encoder_y.transpose(1, 2).reshape(batch_size, -1)

        if self.use_predict_covariate == True:
            if self.covariate_size > 0:
                encoder_y_x = torch.cat(
                    (
                        encoder_y,
                        encoder_x_t.reshape(batch_size, -1),
                        decoder_x_t.reshape(batch_size, -1),
                    ),
                    1,
                )
        else:
            if self.covariate_size > 0:
                encoder_y_x = torch.cat(
                    (
                        encoder_y,
                        encoder_x_t.reshape(batch_size, -1),
                    ),
                    1,
                )

        specific_basis = torch.tensor(specific_basis, device=encoder_y.device, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
        basis = F.interpolate(specific_basis, self.context_length+self.prediction_length, mode='linear', align_corners=True).permute(0, 2, 1)
        encoder_expansion_coefficient_backcast = self.layers_backcast(encoder_y_x).unsqueeze(-1)
        encoder_expansion_coefficient_forecast = self.layers_forecast(encoder_y_x).unsqueeze(-1)

        backcast = torch.matmul(basis[0, :self.context_length, :], encoder_expansion_coefficient_backcast)
        forecast = torch.matmul(basis[0, self.context_length:, :], encoder_expansion_coefficient_forecast)

        return backcast, forecast, encoder_expansion_coefficient_backcast, encoder_expansion_coefficient_forecast

class NBEATSModule(nn.Module):

    def __init__(
        self,
        specific_basis,
        context_length,
        prediction_length,
        output_size: int,
        covariate_size,
        n_blocks: Union[int, List[int]],
        n_layers: list,
        hidden_size: list,
        pooling_sizes: list,
        downsample_frequencies: list,
        pooling_mode,
        interpolation_mode,
        dropout,
        activation,
        initialization,
        batch_normalization,
        shared_weights,
        naive_level: bool,
        use_predict_covariate: bool,
        n_delays: int,
        svd_low_rank: int,
    ):
        super().__init__()

        self.specific_basis = specific_basis[:, :svd_low_rank]
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.output_size = output_size
        self.naive_level = naive_level
        self.svd_low_rank = int(svd_low_rank/len(n_blocks))

        Specific_blocks = self.create_stack(
            n_blocks=n_blocks,
            context_length=context_length,
            prediction_length=prediction_length,
            output_size=output_size,
            covariate_size=covariate_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            pooling_sizes=pooling_sizes,
            downsample_frequencies=downsample_frequencies,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            batch_normalization=batch_normalization,
            dropout=dropout,
            activation=activation,
            shared_weights=shared_weights,
            initialization=initialization,
            use_predict_covariate=use_predict_covariate,
            n_delays=n_delays,
            svd_low_rank=int(svd_low_rank/len(n_blocks)),
        )
        self.Specific_blocks = torch.nn.ModuleList(Specific_blocks)
        self.Weight_layer = nn.Linear(
                in_features=len(n_blocks),
                out_features=1,
            )

    def create_stack(
        self,
        n_blocks,
        context_length,
        prediction_length,
        output_size,
        covariate_size,
        n_layers,
        hidden_size,
        pooling_sizes,
        downsample_frequencies,
        pooling_mode,
        interpolation_mode,
        batch_normalization,
        dropout,
        activation,
        shared_weights,
        initialization,
        use_predict_covariate,
        n_delays,
        svd_low_rank,
    ):
        block_list = []

        for i in range(len(n_blocks)):
            for block_id in range(n_blocks[i]):

                if (len(block_list) == 0) and (batch_normalization):
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False

                if shared_weights and block_id > 0:
                    specific_block = block_list[-1]
                else:
                    n_theta = max(prediction_length // downsample_frequencies[i], 1)

                    specific_block = NEATSBlock(
                        context_length=context_length,
                        prediction_length=prediction_length,
                        output_size=output_size,
                        covariate_size=covariate_size,
                        n_theta=n_theta,
                        hidden_size=hidden_size[i],
                        pooling_sizes=pooling_sizes[i],
                        pooling_mode=pooling_mode,
                        n_layers=n_layers[i],
                        batch_normalization=batch_normalization_block,
                        dropout=dropout,
                        activation=activation,
                        use_predict_covariate=use_predict_covariate,
                        n_delays=n_delays,
                        svd_low_rank=svd_low_rank
                    )

                init_function = partial(init_weights, initialization=initialization)
                specific_block.apply(init_function)
                block_list.append(specific_block)

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

        level = encoder_y[:, -1:].repeat(1, self.prediction_length, 1)
        forecast_level = level.repeat_interleave(torch.tensor(self.output_size, device=level.device), dim=2)

        if self.naive_level:
            block_forecasts = [forecast_level]
            block_backcasts = [encoder_y[:, -1:].repeat(1, self.context_length, 1)]

            forecast = block_forecasts[0]
        else:
            block_forecasts = []
            block_backcasts = []
            block_coefficient_forecasts = []
            block_coefficient_backcasts = []
            forecast = torch.zeros_like(forecast_level, device=forecast_level.device)
        i = 0
        for block in self.Specific_blocks:
            block_backcast, block_forecast, block_coefficient_backcast, block_coefficient_forecast = block(
                encoder_y=residuals, encoder_x_t=encoder_x_t, decoder_x_t=decoder_x_t, specific_basis=self.specific_basis[:, i*self.svd_low_rank:(i+1)*self.svd_low_rank],
            )
            i = i+1

            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)
            block_backcasts.append(block_backcast)
            block_coefficient_forecasts.append(block_coefficient_forecast)
            block_coefficient_backcasts.append(block_coefficient_backcast)
            residuals = residuals - block_backcast

        block_forecasts = torch.stack(block_forecasts, dim=-1)
        block_backcasts = torch.stack(block_backcasts, dim=-1)
        block_coefficient_forecasts = torch.stack(block_coefficient_forecasts, dim=-1)
        block_coefficient_backcasts = torch.stack(block_coefficient_backcasts, dim=-1)

        backcast = residuals
        forecast1 = self.Weight_layer(block_forecasts).squeeze(-1)

        return forecast1, backcast, block_forecasts, block_backcasts, block_coefficient_forecasts, block_coefficient_backcasts


class SSBEM_B(nn.Module):
    def __init__(
        self,
        specific_basis,
        output_size: Union[int, List[int]] = 1,
        context_length: int = 1,
        covariate_size: int = 1,
        prediction_length: int = 1,
        naive_level: bool = False,
        shared_weights: bool = True,
        activation: str = "ReLU",
        initialization: str = "lecun_normal",
        n_blocks: List[int] = [1, 1, 1],
        n_layers: Union[int, List[int]] = 2,
        hidden_size: int = 512,
        pooling_sizes: Optional[List[int]] = None,
        downsample_frequencies: Optional[List[int]] = None,
        pooling_mode: str = "max",
        interpolation_mode: str = "linear",
        batch_normalization: bool = False,
        dropout: float = 0.0,
        use_predict_covariate: bool = False,
        n_delays: int = 10,
        svd_low_rank: int = 10
    ):
        self.output_size = output_size
        self.covariate_size = covariate_size
        self.prediction_length = prediction_length

        if activation == "SELU":
            initialization = "lecun_normal"

        n_stacks = len(n_blocks)
        if pooling_sizes is None:
            pooling_sizes = np.exp2(np.round(np.linspace(0.49, np.log2(prediction_length / 2), n_stacks)))
            pooling_sizes = [int(x) for x in pooling_sizes[::-1]]
        if downsample_frequencies is None:
            downsample_frequencies = [min(prediction_length, int(np.power(x, 1.5))) for x in pooling_sizes]

        # set layers
        if isinstance(n_layers, int):
            n_layers = [n_layers] * n_stacks

        super(SSBEM_B, self).__init__()

        self.model_specific = NBEATSModule(
            specific_basis=specific_basis,
            context_length=context_length,
            prediction_length=prediction_length,
            output_size=to_list(output_size),
            covariate_size=covariate_size,
            n_blocks=n_blocks,
            n_layers=n_layers,
            hidden_size=n_stacks * [2 * [hidden_size]],
            pooling_sizes=pooling_sizes,
            downsample_frequencies=downsample_frequencies,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            dropout=dropout,
            activation=activation,
            initialization=initialization,
            batch_normalization=batch_normalization,
            shared_weights=shared_weights,
            naive_level=naive_level,
            use_predict_covariate=use_predict_covariate,
            n_delays=n_delays,
            svd_low_rank=svd_low_rank
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[Any, Any, Tuple[Any, ...], Tuple[Any, ...],  Tuple[Any, ...],  Tuple[Any, ...], Tuple[Any, ...]]:
        if self.covariate_size > 0:
            encoder_x_t = x[:, :-self.prediction_length, :]
            decoder_x_t = x[:, -self.prediction_length:, :]
        else:
            encoder_x_t = None
            decoder_x_t = None

        encoder_y = y
        residual_for_specific = encoder_y

        forecast_specific, backcast_specific, block_forecasts_specific, block_backcasts_specific, block_coefficient_forecasts, block_coefficient_backcasts = self.model_specific(
            residual_for_specific, encoder_x_t, decoder_x_t
        )

        backcast = encoder_y - backcast_specific
        forecast = forecast_specific

        block_backcasts_specific = block_backcasts_specific.detach()
        block_forecasts_specific = block_forecasts_specific.detach()

        block_coefficient_forecasts = block_coefficient_forecasts.detach()
        block_coefficient_backcasts = block_coefficient_backcasts.detach()

        if isinstance(self.output_size, (tuple, list)):
            forecast = forecast.split(self.output_size, dim=2)
            backcast = backcast.split(1, dim=2)

            block_backcasts_specific = tuple(
                (block.squeeze(3).split(1, dim=2)) for block in block_backcasts_specific.split(1, dim=3)
            )
            block_forecasts_specific = tuple(
                (block.squeeze(3).split(self.output_size, dim=2)) for block in block_forecasts_specific.split(1, dim=3)
            )

            block_coefficient_forecasts = tuple(
                (block.squeeze(3).split(1, dim=2)) for block in block_coefficient_backcasts.split(1, dim=3)
            )
            block_coefficient_backcasts = tuple(
                (block.squeeze(3).split(self.output_size, dim=2)) for block in block_coefficient_forecasts.split(1, dim=3)
            )

        return forecast, backcast, block_backcasts_specific, block_forecasts_specific, block_coefficient_backcasts, block_coefficient_forecasts
