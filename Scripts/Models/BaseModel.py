



class BaseModel():
    def __init__(self,name:str, DataParser: BaseDataParser, LabelParser: BaseLabelParser, save_dir='saved_model'):
        self.name = name
        self.DataParser = DataParser
        self.LabelParser = LabelParser
        self.env_pred = False
        self.model, self.predict_model = self.create_model()
        
        assert self.model is not None, "模型初始化失败"
        self.main_save_dir = save_dir
        self.cur_save_dir = None
        self.load_weight_path = None

    def create_model():
        return model,predict_model