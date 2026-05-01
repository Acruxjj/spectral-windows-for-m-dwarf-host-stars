from math import ceil
from typing import Sequence

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics


# Wrappers de Dropout que continuam ativos na inferência (Monte Carlo Dropout):
# permitem estimar incerteza fazendo várias predições por amostra.
class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


class SpatialMCDropout(tf.keras.layers.SpatialDropout1D):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


class HyperTuningBayCNN:
    """
    CNN 1D com busca Bayesiana de hiperparâmetros (Keras Tuner) e MC Dropout.
    """

    def __init__(
        self,
        input_shape,
        output_units=1,
        outputs_list: Sequence[str] | None = None,
        tuner_directory: str = "hyperparam_search",
        project_name: str = "cnn_bay_tuning",
    ):
        self.input_shape = input_shape
        self.output_units = output_units
        self.outputs_list = list(outputs_list or [])
        self.tuner_directory = tuner_directory
        self.project_name = project_name
        self.best_hps = None
        self.best_model = None
        self.history = None

    def _units_for_output(self, output_name: str) -> int:
        if isinstance(self.output_units, dict):
            return int(self.output_units.get(output_name, 1))
        try:
            return int(self.output_units)
        except Exception:
            return 1

    def build_model(self, hp):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs

        # Blocos convolucionais
        conv_blocks = hp.Int("conv_blocks", 1, 3)
        for b in range(conv_blocks):
            x = tf.keras.layers.Conv1D(
                filters=hp.Int(f"filters_b{b}_a", 16, 128, step=16),
                kernel_size=hp.Choice(f"kernel_b{b}_a_ks", values=[3, 5, 7, 9]),
                padding="same",
                activation="swish",
            )(x)
            x = tf.keras.layers.Conv1D(
                filters=hp.Int(f"filters_b{b}_b", 16, 128, step=16),
                kernel_size=hp.Choice(f"kernel_b{b}_b_ks", values=[3, 5, 7, 9]),
                padding="same",
                activation="swish",
            )(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = SpatialMCDropout(
                rate=hp.Float(f"dropout_rate_conv_b{b}", 0.0, 0.5, step=0.1)
            )(x)

        x = tf.keras.layers.Flatten()(x)

        # Blocos densos
        dense_blocks = hp.Int("dense_layers", 1, 3)
        for d in range(dense_blocks):
            x = tf.keras.layers.Dense(
                units=hp.Int(f"units_dense_{d}", 64, 512, step=64),
                activation="swish",
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            )(x)
            x = MCDropout(hp.Float(f"dropout_rate_dense_{d}", 0.0, 0.5, step=0.1))(x)

        # Camadas de saída: uma por variável alvo
        outputs = {}
        for output_name in self.outputs_list:
            units = self._units_for_output(output_name)
            outputs[output_name] = tf.keras.layers.Dense(
                units=units,
                activation="linear",
                kernel_regularizer=tf.keras.regularizers.l1(1e-5),
                name=output_name,
            )(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Cosine Decay com restarts para o learning rate
        lr = hp.Float("learning_rate", 1e-5, 1e-2, sampling="log")
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=lr,
            first_decay_steps=10,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.1,
        )

        losses = {name: "mse" for name in self.outputs_list}
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=losses
        )
        return model

    def search(
        self,
        train_dataset,
        validation_dataset,
        max_trials: int = 50,
        search_epochs: int = 3,
    ):
        tuner = kt.BayesianOptimization(
            self.build_model,
            objective="val_loss",
            max_trials=max_trials,
            directory=self.tuner_directory,
            project_name=self.project_name,
        )
        tuner.search(
            train_dataset, validation_data=validation_dataset, epochs=search_epochs
        )
        self.best_hps = tuner.get_best_hyperparameters(1)[0]
        return self.best_hps

    def fit_best_model(
        self, train_dataset, validation_dataset, epochs: int = 100, patience: int = 5
    ):
        if not self.best_hps:
            raise ValueError("No best hyperparameters found. Run 'search' first.")

        self.best_model = self.build_model(self.best_hps)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
        self.history = self.best_model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            callbacks=[early_stopping],
            verbose=1,
        )
        return self.best_model, self.history

    def evaluate(
        self, test_dataset, feature_names, y_scalers, plot=True, additional_metrics=True
    ):
        if self.best_model is None:
            raise ValueError("No trained model. Run 'fit_best_model' first.")

        y_true_dict = {name: [] for name in self.outputs_list}
        y_pred_dict = {name: [] for name in self.outputs_list}

        for X_batch, y_batch in test_dataset:
            preds = self.best_model.predict(X_batch, verbose=0)
            if isinstance(preds, dict):
                pred_batch = preds
            elif isinstance(preds, (list, tuple)):
                pred_batch = {
                    name: preds[idx] for idx, name in enumerate(self.outputs_list)
                }
            else:
                pred_batch = {self.outputs_list[0]: preds}

            for name in self.outputs_list:
                y_true_arr = (
                    y_batch[name].numpy()
                    if tf.is_tensor(y_batch[name])
                    else np.array(y_batch[name])
                )
                y_pred_arr = pred_batch[name]
                y_true_dict[name].extend(y_true_arr.reshape(-1).tolist())
                y_pred_dict[name].extend(np.asarray(y_pred_arr).reshape(-1).tolist())

        y_true = np.column_stack([np.array(y_true_dict[n]) for n in self.outputs_list])
        y_pred = np.column_stack([np.array(y_pred_dict[n]) for n in self.outputs_list])

        y_true_inv = np.column_stack(
            [
                y_scalers[i].inverse_transform(y_true[:, i].reshape(-1, 1)).ravel()
                for i in range(len(self.outputs_list))
            ]
        )
        y_pred_inv = np.column_stack(
            [
                y_scalers[i].inverse_transform(y_pred[:, i].reshape(-1, 1)).ravel()
                for i in range(len(self.outputs_list))
            ]
        )

        return self._evaluate_outputs(
            y_true_inv, y_pred_inv, feature_names, plot, additional_metrics
        )

    def _evaluate_outputs(
        self, y_true, y_pred, feature_names, plot=True, show=True, save_path=None
    ):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        num_outputs = y_true.shape[1]

        rows = []
        for i in range(num_outputs):
            yt = y_true[:, i]
            yp = y_pred[:, i]
            r2 = metrics.r2_score(yt, yp)
            mae = metrics.mean_absolute_error(yt, yp)
            mse = metrics.mean_squared_error(yt, yp)
            rmse = np.sqrt(mse)
            rows.append(
                {
                    "R2": round(r2, 3),
                    "MAE": round(mae, 3),
                    "MSE": round(mse, 3),
                    "RMSE": round(rmse, 3),
                }
            )

        results_df = pd.DataFrame(rows, index=feature_names)

        fig_axs = None
        if plot:
            num_cols = min(num_outputs, 3)
            num_rows = max(1, ceil(num_outputs / num_cols))
            fig, axs = plt.subplots(
                num_rows,
                num_cols,
                figsize=(5 * num_cols, 5 * num_rows),
                squeeze=False,
            )
            axs_flat = axs.ravel()

            for i in range(num_outputs):
                ax = axs_flat[i]
                yt = y_true[:, i]
                yp = y_pred[:, i]

                # density coloring via 2-D histogram
                h, xedges, yedges = np.histogram2d(yt, yp, bins=40)
                xi = np.clip(np.digitize(yt, xedges[:-1]) - 1, 0, h.shape[0] - 1)
                yi = np.clip(np.digitize(yp, yedges[:-1]) - 1, 0, h.shape[1] - 1)
                density = h[xi, yi]
                order = density.argsort()

                sc = ax.scatter(
                    yt[order],
                    yp[order],
                    c=density[order],
                    cmap="plasma",
                    s=18,
                    alpha=0.85,
                    linewidths=0,
                )
                cbar = fig.colorbar(sc, ax=ax, pad=0.02)
                cbar.set_label("Density", fontsize=11)
                cbar.ax.tick_params(labelsize=9)

                # symmetric 1:1 limits with margin
                lo = min(yt.min(), yp.min())
                hi = max(yt.max(), yp.max())
                margin = 0.04 * (hi - lo)
                lims = (lo - margin, hi + margin)
                ax.plot(lims, lims, color="crimson", lw=1.5, ls="--", zorder=3)
                ax.set_xlim(lims)
                ax.set_ylim(lims)
                ax.set_aspect("equal")

                # metrics annotation
                r2 = results_df.loc[feature_names[i], "R2"]
                mae = results_df.loc[feature_names[i], "MAE"]
                rmse = results_df.loc[feature_names[i], "RMSE"]
                stats_text = f"$R^2={r2}$\nMAE$={mae}$\nRMSE$={rmse}$"
                ax.text(
                    0.05,
                    0.95,
                    stats_text,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=11,
                    bbox=dict(
                        boxstyle="round,pad=0.4", fc="white", ec="none", alpha=0.75
                    ),
                )

                ax.set_xlabel(f"True {feature_names[i]}")
                ax.set_ylabel(f"Predicted {feature_names[i]}")
                ax.set_title(feature_names[i], fontweight="bold")

            for i in range(num_outputs, num_rows * num_cols):
                fig.delaxes(axs_flat[i])

            fig.tight_layout()

            if save_path:
                import os

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            fig_axs = (fig, axs_flat)

        return (results_df, fig_axs) if plot else (results_df, None)

    def plot_losses(self, show: bool = True, save_path: str | None = None):
        if self.history is None:
            raise ValueError("No training history found. Train the model first.")

        train_loss = self.history.history.get("loss", [])
        val_loss = self.history.history.get("val_loss", [])
        epochs = np.arange(1, len(train_loss) + 1)

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(epochs, train_loss, color="steelblue", lw=2, label="Training")
        ax.plot(epochs, val_loss, color="darkorange", lw=2, label="Validation")

        ax.fill_between(epochs, train_loss, val_loss, color="steelblue", alpha=0.08)

        best_epoch = int(np.argmin(val_loss)) + 1
        ax.axvline(best_epoch, color="gray", ls=":", lw=1.2)
        ax.annotate(
            f"best epoch {best_epoch}",
            xy=(best_epoch, val_loss[best_epoch - 1]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=10,
            color="gray",
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.legend()
        fig.tight_layout()

        if save_path:
            import os

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig, ax

    def summary(self):
        if self.best_model is None:
            print("No model is built yet. Train or load a model first.")
        else:
            self.best_model.summary()
