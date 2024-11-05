import torch

class AttentionMap():
    def __init__(self):
        from translator import SeamlessTranslator
        from config import M4TLargeTranslatorConfig
        args = M4TLargeTranslatorConfig
        self.st = SeamlessTranslator(args)
    
    def get_input_ids(self, in_text):
        input_ids = self.st.processor(
            text = in_text,
            src_lang = self.st.args.lang_dict['English'],
            return_tensors="pt"
        ).input_ids
        return input_ids
    
    def get_gen_outputs(self, input_ids, tgt_lang='Chinese'):
        gen_outputs = self.st.model.generate(
            input_ids.to(self.st.device),
            tgt_lang=self.st.args.lang_dict[tgt_lang],
            generate_speech=False,
            output_attentions=True
        )
        return gen_outputs
    
    def get_att(self, gen_outputs):
        # Assuming outputs.cross_attentions is the list obtained from the generate() function
        cross_attentions = gen_outputs.cross_attentions  # This is of shape [sequence_length][n_layers][batch_size, n_heads, 1 or 2, input_sequence_length]

        sequence_length = len(cross_attentions)
        n_layers = len(cross_attentions[0])
        batch_size, n_heads, _, input_sequence_length = cross_attentions[0][0].shape

        # Initialize a list to store reshaped cross-attentions
        reshaped_cross_attentions = [[] for _ in range(n_layers)]

        # Iterate over layers
        for layer in range(n_layers):
            # Collect attention matrices for the current layer across all sequence lengths
            layer_attentions = []
            for seq in range(sequence_length):
                if seq == 0:
                    layer_attentions = list(torch.split(cross_attentions[seq][layer], 1, dim=2))
                else:
                    layer_attentions.append(cross_attentions[seq][layer])
            
            # Stack along the new axis (decoder sequence length)
            stacked_attentions = torch.cat(layer_attentions, dim=2)  # Shape: [batch_size, n_heads, sequence_length, input_sequence_length]
            
            # Append to the reshaped list
            reshaped_cross_attentions[layer] = stacked_attentions

        # Now reshaped_cross_attentions is a list of length n_layers
        # Each element has shape [batch_size, n_heads, sequence_length, input_sequence_length]
        return reshaped_cross_attentions
    
    def plot(self, input_ids, gen_outputs, att):
        encoder_text = self.st.processor.tokenizer.convert_ids_to_tokens(input_ids[0])
        decoder_text = self.st.processor.tokenizer.convert_ids_to_tokens(gen_outputs['sequences'][0][1:])
        # https://nlp.seas.harvard.edu/2018/04/03/attention.html
        import seaborn as sns
        from matplotlib import pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['font.family'] = ['SimHei']
        def draw(data, x, y, ax):
            sns.heatmap(data, 
                xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                cbar=False, ax=ax)
        fig, axs = plt.subplots(1, 1)
        
        # draw(att[-2][0, -1, :, :].cpu(), encoder_text, decoder_text, axs)
        
        att = att[-2]
        att = torch.mean(att, dim=1)[0]
        draw(att.cpu(), encoder_text, decoder_text, axs)
    
    def get_top_k(self, att, encoder_text, decoder_text, encoder_subset, top_k=3):
        att = att.transpose()
        res = []
        for item_i in encoder_subset:
            for idx, item_j in enumerate(encoder_text):
                if item_i == item_j:
                    top_vals, top_indices = torch.topk(att[idx], k=top_k)
                    vals = []
                    for i in top_indices:
                        vals.append(decoder_text[i])
                    res.append(item_i, vals)
                    break
        return res