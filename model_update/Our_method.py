from data_factory.data_loader import OnlineSILoader
from utils.utils import *

def str2bool(v):
    return v.lower() in ('true')

class Online_regressor(object):

    def __init__(self, base_estimator, model_load_path, device):
        super(Online_regressor, self).__init__()
        self.base_estimator = base_estimator
        self.model_load_path = model_load_path
        self.device = device
        if torch.cuda.is_available():
            self.base_estimator.cuda()

    def recursive_update(self, x_1, y_1, x_pt, y_pt, y, W_old, Q_old, lr, dataset):

        self.base_estimator.load_state_dict(
            torch.load(
                os.path.join(str(self.model_load_path), str(dataset) + '_checkpoint.pth')))
        self.base_estimator.eval()

        X_new = self.base_estimator(x_pt, y_pt)[-3].cpu().detach().numpy().squeeze()[0, :][:, None]
        X_old = self.base_estimator(x_1, y_1)[-3].cpu().detach().numpy().squeeze()[0, :][:, None]

        X_new = np.vstack([X_new, [[1]]])
        X_old = np.vstack([X_old, [[1]]])

        # prepare to recursive update W_new
        P_new = Q_old - (Q_old @ X_new @ X_new.T @ Q_old) / (1 + X_new.T @ Q_old @ X_new)
        Q_new = P_new + (P_new @ X_old @ X_old.T @ P_new) / (1 - X_old.T @ P_new @ X_old)
        y = y.cpu().detach().numpy().squeeze()
        y_pt = y_pt.cpu().detach().numpy().squeeze()

        W_new = W_old.T + lr * Q_new @ (X_new @ (y[-1] - X_new.T @ W_old.T) - (X_old @ (y_pt[0] - X_old.T @ W_old.T)))

        W_update = W_new[:-1, :]
        B_update = W_new[-1, :]

        if dataset == "Online_Recursive_0":
            self.base_estimator.model_specific.Weight_layer.bias.data = torch.tensor(B_update, dtype=torch.float32)
        else:
            self.base_estimator.model_specific.Weight_layer.weight.data = torch.tensor(W_update.T, dtype=torch.float32)
            self.base_estimator.model_specific.Weight_layer.bias.data = torch.tensor(B_update, dtype=torch.float32)

        if dataset == "Online_Initialization":
            modelset = "Online_Recursive_0_transition"
            dataset = "Online_Recursive_0"
        elif dataset == "Online_Recursive_0":
            modelset = "Online_Recursive_1_transition"
            dataset = "Online_Recursive_1"

        torch.save(self.base_estimator.state_dict(), os.path.join(self.model_load_path, str(modelset)+'_checkpoint.pth'))

        return Q_new, dataset, modelset

    def predict(self, X, y, modelset, data_path, win_size, step, prediction_length):
        scaler = OnlineSILoader(data_path, win_size, step, prediction_length).scaler
        X = scaler.transform(X)

        X = torch.as_tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device)[:, None].unsqueeze(0)

        self.base_estimator.load_state_dict(
            torch.load(
                os.path.join(str(self.model_load_path), str(modelset) + '_checkpoint.pth')))
        self.base_estimator.eval()
        with torch.no_grad():
            y_new = self.base_estimator(X, y)

        return y_new