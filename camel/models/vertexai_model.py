import vertexai
from vertexai.preview.language_models import TextGenerationModel

class VertexAIModel(BaseModelBackend):
    def __init__(self, model_type: ModelType, model_config_dict: Dict[str, Any]):
        super().__init__(model_type, model_config_dict)
        self.model_name = model_config_dict["model_name"]
        self.project_id = model_config_dict["project_id"]
        self.location = model_config_dict.get("location", "us-central1")
        self.temperature = model_config_dict.get("temperature", 0.2)
        self.max_decode_steps = model_config_dict.get("max_decode_steps", 256)
        self.top_p = model_config_dict.get("top_p", 0.8)
        self.top_k = model_config_dict.get("top_k", 40)

    def run(self, messages: List[Dict]) -> Dict[str, Any]:
        prompt = " ".join(message["content"] for message in messages)
        
        vertexai.init(project=self.project_id, location=self.location)
        model = TextGenerationModel.from_pretrained(self.model_name)
        response = model.predict(
            prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_decode_steps,
            top_k=self.top_k,
            top_p=self.top_p
        )

        return {"choices": [{"message": {"role": "system", "content": response.text}}]}
