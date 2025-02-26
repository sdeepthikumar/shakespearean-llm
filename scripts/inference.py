import torch
from models.transformer import DecoderOnlyTransformer, Config
import tiktoken

def load_model():
    config = Config()
    model = DecoderOnlyTransformer(config)
    model.load_state_dict(torch.load("final_model.pt"))
    model.eval()
    return model

def predict(input_text, max_new_tokens=50):
    model = load_model()
    enc = tiktoken.get_encoding("gpt2")
    input_ids = torch.tensor(enc.encode(input_text), dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get the logits for the current input
            logits = model(input_ids)
            
            # Get the predicted token index (last token in the sequence)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            # Append the predicted token to the input_ids
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            # Optionally, break if an end-of-sequence token is generated
            # if next_token_id.item() == enc.eos_token_id:
            #     break

    # Decode the generated sequence
    predicted_text = enc.decode(input_ids.squeeze().tolist())
    return predicted_text

if __name__ == '__main__':
    input_text = "More learned than the ears--waving thy head, Which often, thus, correcting thy stout heart,"
    predicted_text = predict(input_text, max_new_tokens=50)
    print("Predicted text:", predicted_text) 