"""Request-local shared vision + independent language batches + restricted output head.

Targets the inspected Transformers 5.17 Qwen3.5 implementation. No persistent image cache.
"""
import time

from .backend import QwenVerifier


class FastQwenVerifier(QwenVerifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.batch_size>4:
            raise ValueError('Validated fast-path micro-batch limit is 4; use chunks for larger requests')
        if self.adapter:
            self.model = self.model.merge_and_unload()
        self.model.eval()

    def _score(self, image, request, questions):
        torch = self.torch
        core = self.model.model
        scores = []; preprocessing = 0.0; forward = 0.0
        image_features = None
        for start in range(0, len(questions), self.batch_size):
            chunk = questions[start:start+self.batch_size]
            begin = time.perf_counter()
            inputs = self.prepare(image, request, chunk)
            self._sync(); middle = time.perf_counter()
            with torch.inference_mode():
                grid = inputs['image_grid_thw']
                if len(grid) != len(chunk) or not torch.equal(grid,grid[:1].expand_as(grid)):
                    raise ValueError('Fast path supports exactly one identical screenshot per question')
                if image_features is None:
                    patches = int(grid[0].prod().item())
                    image_features = core.get_image_features(inputs['pixel_values'][:patches],
                        grid[:1], return_dict=True).pooler_output[0]
                embeddings = core.get_input_embeddings()(inputs['input_ids'])
                features = image_features.repeat(len(chunk),1).to(embeddings.dtype)
                mask,_ = core.get_placeholder_mask(inputs['input_ids'],inputs_embeds=embeddings,
                                                    image_features=features)
                embeddings = embeddings.masked_scatter(mask,features)
                positions = core.compute_3d_position_ids(input_ids=inputs['input_ids'],
                    image_grid_thw=grid, inputs_embeds=embeddings, attention_mask=inputs['attention_mask'],
                    mm_token_type_ids=inputs.get('mm_token_type_ids'))
                hidden = core.language_model(inputs_embeds=embeddings,position_ids=positions,
                    attention_mask=inputs['attention_mask'],use_cache=False,return_dict=True).last_hidden_state[:,-1,:]
                # Only the candidate rows are needed; no 248k-way vocabulary projection.
                weight = self.model.lm_head.weight[self.label_ids]
                bias = self.model.lm_head.bias
                result = torch.nn.functional.linear(hidden,weight,
                    bias[self.label_ids] if bias is not None else None).float().cpu().tolist()
            self._sync(); end = time.perf_counter()
            scores.extend(row[:len(q.options)] for row,q in zip(result,chunk))
            preprocessing += middle-begin; forward += end-middle
        return scores,preprocessing,forward

    def verify(self,image,request):
        result=super().verify(image,request)
        result.metadata.update(shared_visual_encoding=len(request.questions)>1, shared_prefix_cache=False,
            backend='shared-vision-independent-language-batch', adapter_merged=bool(self.adapter),
            restricted_output_head=True,
            visual_encoder_calls=2 if request.check_order else 1)
        return result
