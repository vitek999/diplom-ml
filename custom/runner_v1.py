import os

from Handler import Dataset, f1_metric, make_dataset, save_plot, save_plot_one_field
import tensorflow as tf

from models_tf import mlp_net_tf_v2, cnn_net_tf, mlp_net_tf, cnn_net_tf_v2, gru_net_tf, lstm_net_tf, gru_net_big_tf

if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))
    window_time = 0.5
    print(f"Start, window_time = {window_time}")

    print("load dataset")
    train_ds = Dataset(is_train=True, window_time=window_time)
    test_ds = Dataset(is_train=False, window_time=window_time)

    print("load to Tensor")

    train_kds = make_dataset(train_ds.x, train_ds.y, True)
    test_kds = make_dataset(test_ds.x, test_ds.y, False)

    print("Create model")
    model_list = [mlp_net_tf_v2(train_ds.x), cnn_net_tf_v2(train_ds.x), gru_net_tf(train_ds.x), lstm_net_tf(train_ds.x), gru_net_big_tf(train_ds.x)]
    # model_list = [mlp_net_tf(train_ds.x_feature)]

    for model in model_list:
        model.compile(optimizer="adam", loss="categorical_crossentropy",
                      metrics=["AUC", f1_metric, "Recall", "Precision", "accuracy"])

        print("Start fit")
        history = model.fit(train_kds, validation_data=test_kds, epochs=20)

        print("Predict")
        print("=" * 50)
        print(model.predict(test_ds.x))
        print("=" * 50)

        print("Generate plot")
        metric_list = ["loss", "auc", "f1_metric", "accuracy", "precision", "recall"]
        for metric in metric_list:
            script_dir = os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, 'res_mlp/')

            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            save_plot_one_field(history.history, metric, metric, f"{results_dir}{model.name}_{metric}.png")
            print(f"Save {metric}")
