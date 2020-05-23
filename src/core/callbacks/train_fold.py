from .base import Callback, CallbackOrder


# on_train_fold_end
class CalcFoldRMSLECallback(Callback):
    signature = ""
    callback_order = CallbackOrder.HIGHER
