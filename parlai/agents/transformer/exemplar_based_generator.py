import os
from pathlib import Path
from typing import List

import faiss
import numpy as np
import torch
from tqdm import tqdm

import parlai.utils.logging as logging
from parlai.agents.transformer.modules import (TransformerGeneratorModel,
                                               TransformerMemNetModel)
from parlai.utils.io import PathManager
from parlai.utils.torch import padded_tensor


class ExemplarBasedGeneratorModule(torch.nn.Module):
    def __init__(
        self,
        opt,
        retriever_opt,
        retriever_dict,
        generator_opt,
        generator_dict,
        fixed_candidates,
        fixed_candidate_retriever_vecs,
        fixed_candidate_generator_vecs,
    ):
        super().__init__()
        self.opt = opt
        self.retriever_dict = retriever_dict
        self.generator_dict = generator_dict
        self.retriever = TransformerMemNetModel(retriever_opt,
                                                retriever_dict)

        # use n_segments = 2 to distinguish context and exemplar
        logging.info(f"Use n_segments = {self.opt['n_segments']}")
        generator_opt["n_segments"] = self.opt["n_segments"]
        self.generator = TransformerGeneratorModel(generator_opt,
                                                   generator_dict)

        if opt["init_corge_training"]:
            self.init_corge_training()

        if self.opt["freeze_retriever"]:
            logging.info("Freeze retriever during training.")
            self._freeze_retriever()

        self.fixed_candidates: List[str] = fixed_candidates
        self.fixed_candidate_retriever_vecs: torch.LongTensor = \
            padded_tensor(
                fixed_candidate_retriever_vecs,
                pad_idx=self.retriever_dict._word_lookup(self.retriever_dict.null_token),
                use_cuda=False,
                fp16friendly=self.opt.get('fp16', False),
            )[0]
        self.fixed_candidate_generator_vecs: torch.LongTensor = \
            padded_tensor(
                fixed_candidate_generator_vecs,
                pad_idx=self.generator_dict._word_lookup(self.generator_dict.null_token),
                use_cuda=False,
                fp16friendly=self.opt.get('fp16', False),
            )[0]

        assert len(self.fixed_candidates) == self.fixed_candidate_retriever_vecs.size(0)
        assert len(self.fixed_candidates) == self.fixed_candidate_generator_vecs.size(0)

    def _freeze_retriever(self):
        # Freeze retriever's whole parameters following corge.
        for parameter in self.retriever.parameters():
            parameter.requires_grad = False

    def _load_state_dict(self, path):
        import parlai.utils.pickle

        with PathManager.open(path, 'rb') as f:
            states = torch.load(
                f, map_location=lambda cpu, _: cpu, pickle_module=parlai.utils.pickle
            )
        return states["model"]

    def load_retriever(self):
        retriever_state_dict = self._load_state_dict(self.opt["retriever_model_path"])
        self.retriever.load_state_dict(retriever_state_dict)

    def load_generator(self):
        generator_state_dict = self._load_state_dict(self.opt["generator_model_path"])
        # Here, the positional encoding of the generator should be changed.
        # So, need to initialize position embeddings since we use different n_length from the
        # pre-trained generator.
        generator_state_dict.pop("encoder.position_embeddings.weight", None)
        generator_state_dict.pop("decoder.position_embeddings.weight", None)
        self.generator.load_state_dict(generator_state_dict, strict=False)

    def init_corge_training(self):
        logging.info("Loading initial model for training")
        self.load_retriever()
        self.load_generator()

    def _score(self, output, cands):
        if cands.dim() == 2:
            return torch.matmul(output, cands.t())
        elif cands.dim() == 3:
            return torch.bmm(output.unsqueeze(1), cands.transpose(1, 2)).squeeze(1)
        else:
            raise RuntimeError(
                'Unexpected candidate dimensions {}' ''.format(cands.dim())
            )

    def score_candidates(self, xs, padded_cands):
        """
        Encode candidates.

        :param LongTensor[batch,seqlen] xs: input tokens IDs
        :param LongTensor[batch,num_cands,seqlen] cands: candidate token IDs
        """
        context_h, cands_h = self.retriever(xs=xs, mems=None, cands=padded_cands)
        scores = self._score(context_h, cands_h)

        return scores

    def prepare_faiss_index(self, opt):
        # loading candidate encs
        if opt["load_candidate_encs"] and Path(opt["load_candidate_encs_path"]).exists():
            logging.info("Load candidate encs from "
                         f"{os.path.abspath(opt['load_candidate_encs_path'])}.")
            fixed_candidate_encs = np.load(opt["load_candidate_encs_path"])
        else:
            fixed_candidate_encs = self.get_candidate_encs(
                self.fixed_candidate_retriever_vecs,
            )
            if opt['save_candidate_encs_path']:
                logging.info("Save candidate encs to "
                             f"{os.path.abspath(opt['save_candidate_encs_path'])}.")
                np.save(opt["save_candidate_encs_path"], fixed_candidate_encs)

        # sanity check; whether the number of vecs are not different
        assert len(fixed_candidate_encs) == len(self.fixed_candidate_retriever_vecs)
        assert len(fixed_candidate_encs) == len(self.fixed_candidate_generator_vecs)

        # Use fp32 for faiss
        fixed_candidate_encs = fixed_candidate_encs.astype(np.float32)

        # build Faiss index for retriever
        self.build_faiss_index(fixed_candidate_encs)

    def get_candidate_encs(self, vecs: torch.LongTensor) -> np.ndarray:
        """
        Encode candidate into vector using candidate encoder of the retriever.
        """
        vec_batches = [vecs[i: i + 512] for i in range(0, len(vecs), 512)]
        cand_encs = []
        logging.info(
            f"Encoding fixed candidate set ({len(vec_batches)} batch(es) of up to 512)"
        )
        self.retriever.cand_encoder.eval()
        with torch.no_grad():
            for batch in tqdm(vec_batches):
                # each element in batch has shape of [seqlen]
                cands_h = self.retriever.encode_cand(batch.cuda())  # [bsz, seqlen]
                assert cands_h.dim() == 2
                cand_encs.append(cands_h.cpu())
        cand_encs_np = torch.cat(cand_encs, dim=0).data.numpy()  # [total_num, seqlen]
        logging.info("Finished encoding.")
        return cand_encs_np

    def build_faiss_index(self, fixed_candidate_vecs: np.ndarray):
        num_embeddings, index_dim = fixed_candidate_vecs.shape
        cpu_faiss_index = faiss.IndexFlatIP(index_dim)
        self.faiss_index = faiss.index_cpu_to_all_gpus(cpu_faiss_index)
        self.faiss_index.add(fixed_candidate_vecs)

    @property
    def null_token(self):
        return torch.LongTensor([self.generator_dict._word_lookup(self.generator_dict.null_token)])

    def _calculate_same_response(self, retrieved_candidates_vecs, label_vec, thr=0.6):
        """
        Slow heuristic to calculate the jaccard distance :(
        Assume that the response is same if two vectors have jaccard sim. >= thresh
        """
        _retrieved_candidates_vecs = retrieved_candidates_vecs.numpy().tolist()
        _label_vec = label_vec.numpy().tolist()
        null_idx = self.null_token.data[0].item()

        _label_vec_set = set(_label_vec)
        try:
            _label_vec_set.remove(null_idx)
        except KeyError:
            pass

        jaccard_sims = []
        for _retrieved_candidate_vec in _retrieved_candidates_vecs:
            _retrieved_candidate_set = set(_retrieved_candidate_vec)
            try:
                _retrieved_candidate_set.remove(null_idx)
            except KeyError:
                pass

            intersection = len(_retrieved_candidate_set & _label_vec_set)
            union = len(_retrieved_candidate_set | _label_vec_set)
            jaccard_sims.append(intersection / union)

        return torch.FloatTensor(jaccard_sims) >= thr

    def retrieve_topk_candidates(self, batch, topk: int, use_cand_encoder: bool = True):
        bsz = batch.retriever_text_vec.size(0)
        device = batch.retriever_text_vec.device
        _, context_h = self.retriever.encode_context_memory(
            batch.retriever_text_vec, memories_w=None, context_segments=None,
        )  # [bsz, emb_dims]

        if use_cand_encoder:
            # k-Nearest Exemplar
            recognize_cand_h = self.retriever.encode_cand(batch.retriever_label_vec)  # [bsz, emb_dims]
        else:
            # use context embedding as a query
            recognize_cand_h = context_h

        faiss_scores, candidate_indices = self.faiss_index.search(
            recognize_cand_h.data.cpu().numpy().astype(np.float32), topk)  # [bsz, topk]
        candidate_indices_tensor = torch.LongTensor(candidate_indices)

        # if there is no available candidate, then index will have -1 so it should be masked.
        non_effective_indices = candidate_indices_tensor.eq(-1).long()  # [bsz, topk]
        effective_indices = candidate_indices_tensor.ne(-1).long()

        retriever_indices = torch.randint(
            0, len(self.fixed_candidates), size=candidate_indices_tensor.shape,
        ) * non_effective_indices + candidate_indices_tensor * effective_indices

        list_retrieved_topk_candidates = []
        list_generator_topk_inputs = []
        list_chosen_candidates = []  # List[List[str]]
        for instance_idx, retriever_index in enumerate(retriever_indices):
            retrieved_topk_candidates = torch.index_select(
                self.fixed_candidate_retriever_vecs,
                dim=0,
                index=retriever_index,
            )  # [topk, seq_lens]
            generator_topk_inputs = torch.index_select(
                self.fixed_candidate_generator_vecs,
                dim=0,
                index=retriever_index,
            )  # [topk, seq_lens]

            # Heuristic impl:
            # Compare the exemplar with GT label in training, and exclude same
            # utterance by using jaccard distance.
            if self.training:
                compare_len = min(
                    generator_topk_inputs.shape[1],
                    len(batch.label_vec[instance_idx]),
                )
                label_vec = batch.label_vec.to("cpu")
                is_same_response = self._calculate_same_response(
                    generator_topk_inputs[:, :compare_len],
                    label_vec[instance_idx][:compare_len]
                ).long()

                retriever_index = torch.randint(
                    0, len(self.fixed_candidates), size=retriever_index.shape
                ) * is_same_response + retriever_index * (1 - is_same_response)
                retrieved_topk_candidates = torch.index_select(
                    self.fixed_candidate_retriever_vecs,
                    dim=0,
                    index=retriever_index,
                )  # [topk, seq_lens]
                generator_topk_inputs = torch.index_select(
                    self.fixed_candidate_generator_vecs,
                    dim=0,
                    index=retriever_index,
                )  # [topk, seq_lens]

            list_retrieved_topk_candidates.append(retrieved_topk_candidates)
            list_generator_topk_inputs.append(generator_topk_inputs)
            list_chosen_candidates.append([self.fixed_candidates[ind]
                                           for ind in retriever_index])

        all_retrieved_topk_candidates = torch.cat(
            list_retrieved_topk_candidates, dim=0).to(device)  # [topk * bsz, seq_lens]
        all_generator_topk_inputs = torch.cat(
            list_generator_topk_inputs, dim=0).view(bsz, topk, -1).to(device)  # [bsz, topk, seq_lens]

        with torch.no_grad():
            self.retriever.cand_encoder.eval()
            cands_h = self.retriever.encode_cand(
                all_retrieved_topk_candidates).view(bsz, topk, -1)  # [bsz, topk, emb_dims]

        scores = self._score(context_h, cands_h)  # [bsz, topk]

        return all_generator_topk_inputs, scores, list_chosen_candidates

    def prepare_generator_input(self, batch, topk, use_cand_encoder):
        retrieved_topk_candidates, candidate_scores, list_chosen_candidates = \
            self.retrieve_topk_candidates(batch, topk, use_cand_encoder)

        xs = torch.unsqueeze(batch.text_vec, dim=1).repeat(1, topk, 1)  # [bsz, topk, seqlen]
        bsz = xs.size(0)
        generator_input = torch.cat(
            [xs, retrieved_topk_candidates],
            dim=-1,
        )  # [bsz, topk, seqlen + cand_lens]
        generator_input = generator_input.contiguous().view(bsz * topk, -1)
        xs_lengths = batch.text_lengths
        segments = torch.arange(generator_input.size(-1)).view(1, -1).repeat(bsz, 1)
        segments = segments < torch.LongTensor(xs_lengths).view(-1, 1)  # [bsz, seqlen + cand_lens]
        segments = segments.long().to(generator_input.device).view(bsz, 1, -1).repeat(1, topk, 1)
        segments = segments.view(-1, segments.size(-1))
        encoder_states = self.generator.encoder(generator_input,
                                                segments=segments)

        return retrieved_topk_candidates, candidate_scores, list_chosen_candidates, encoder_states

    def forward(self, batch, topk, use_cand_encoder=False):
        """
        :param xs:
            input to the encoder, [bsz, seqlen]
        :param ys:
            expected output from the decoder, used for teacher forcing. [bsz, outlen]
        """
        retrieved_topk_candidates, candidate_scores, list_chosen_candidates, encoder_states = \
            self.prepare_generator_input(batch, topk, use_cand_encoder=use_cand_encoder)

        # use teacher forcing
        ys = batch.label_vec
        bsz = ys.size(0)
        ys = torch.unsqueeze(ys, dim=1).repeat(1, topk, 1)
        out_len = ys.size(-1)
        scores, preds = self.generator.decode_forced(encoder_states,
                                                     ys.contiguous().view(-1, out_len))
        # scores: [bsz x topk, outlen, num_features]
        # preds : [bsz x topk, outlen]

        scores = scores.view(bsz, topk, out_len, -1)
        preds = preds.view(bsz, topk, out_len)

        return scores, candidate_scores, preds, encoder_states, list_chosen_candidates
