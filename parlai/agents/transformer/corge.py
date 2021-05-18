#!/usr/bin/env python3
"""
COnnecting Retriever and Generator for Exemplar-based generation
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

import parlai.utils.logging as logging
from parlai.agents.transformer.exemplar_based_generator import \
    ExemplarBasedGeneratorModule
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.dict import DictionaryAgent
from parlai.core.metrics import AverageMetric, TextMetric
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch, Output
from parlai.core.torch_generator_agent import PPLMetric
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.utils.io import PathManager
from parlai.utils.misc import warn_once
from parlai.utils.torch import argsort, atomic_save, neginf


class CorgeLoss(torch.nn.Module):
    def __init__(self, null_idx, use_fp16=False):
        super().__init__()
        self.NULL_IDX = null_idx
        self.use_fp16 = use_fp16
        self.T = 1

        if not self.use_fp16:
            self.likelihood_loss = torch.nn.CrossEntropyLoss(
                ignore_index=self.NULL_IDX, reduction='none')
        else:
            self.likelihood_loss = FP16SafeCrossEntropy(ignore_index=self.NULL_IDX,
                                                        reduction='none')

    def forward(self, scores, candidate_scores, labels):
        """
        scores: [bsz, num_cands, seqlen, num_features]
        candidate_scores: [bsz, num_cands]
        labels: [bsz, num_cands, seqlen]
        """
        bsz, _ = candidate_scores.shape

        log_likelihood_loss = self.likelihood_loss(
            scores.permute(0, 3, 1, 2), labels,
        )  # [bsz, num_cands, seqlen]
        log_likelihood = -log_likelihood_loss
        cand_log_softmax = torch.log_softmax(candidate_scores / self.T, dim=-1)  # [bsz, num_cands]
        marginalized_logprobs = torch.cat([
            log_likelihood,
            torch.unsqueeze(cand_log_softmax, dim=-1),
        ], dim=-1).sum(-1)  # [bsz, num_cands]
        final_loss = -torch.logsumexp(marginalized_logprobs, dim=-1)  # [bsz]
        log_likelihood_loss = log_likelihood_loss.sum(-1)

        return final_loss, log_likelihood_loss


class RetnrefLoss(torch.nn.Module):
    def __init__(self, null_idx, use_fp16=False):
        super().__init__()
        self.NULL_IDX = null_idx
        self.use_fp16 = use_fp16

        if not self.use_fp16:
            self.likelihood_loss = torch.nn.CrossEntropyLoss(
                ignore_index=self.NULL_IDX, reduction='none')
        else:
            self.likelihood_loss = FP16SafeCrossEntropy(ignore_index=self.NULL_IDX,
                                                        reduction='none')

    def forward(self, scores, candidate_scores, labels):
        """
        scores: [bsz, num_cands, seqlen, num_features]
        candidate_scores: [bsz, num_cands]
        labels: [bsz, num_cands, seqlen]
        """
        bsz, _ = candidate_scores.shape

        log_likelihood_loss = self.likelihood_loss(
            scores.permute(0, 3, 1, 2), labels,
        )  # [bsz, num_cands, seqlen]
        log_likelihood_loss = log_likelihood_loss.sum(-1)
        final_loss = log_likelihood_loss.sum(-1)

        return final_loss, log_likelihood_loss


class CorgeAgent(TransformerGeneratorAgent):

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        agent = parser.add_argument_group('Torch Corge Agent')
        agent.add_argument('--retriever-model-path', type=str,
                           help='Opt path for retriever model')
        agent.add_argument('--generator-model-path', type=str,
                           help='Opt path for generator model')
        agent.add_argument('--init-corge-training',
                           dest='init_corge_training',
                           default=False,
                           action='store_true',
                           help='When trying to initialize exemplar-based generation model, '
                                'we load the pre-trained model from vanilla generator and retriever.')
        agent.add_argument('--corge-topk-cands', type=int, default=2,
                           help='How many candidates will be used while training exemplar-based model.')
        agent.add_argument('--fixed-candidates-path', type=str,
                           help='Where the candidates (text) are stored.')
        agent.add_argument('--freeze-retriever', type=bool, default=False,
                           help='Decide whether we will train retriever model during training or not.')
        agent.add_argument('--use-kne', type=bool, default=True,
                           help='Decide whether to use kNE or not.')
        agent.add_argument('--criterion-type', type=str, choices=['CorgeLoss', 'RetnrefLoss'],
                           default='CorgeLoss')

        # candidate encs mean the encodings for the pre-defined retriever index.
        agent.add_argument('--load-candidate-encs', type=bool, default=True)
        agent.add_argument('--load-candidate-encs-path', type=str)
        agent.add_argument('--save-candidate-encs-path', type=str, default=None)

        # candidate vecs means the vectorized candidates using tokenizer (e.g. BPE dictionary).
        # we use two different vecs because some retrievers and generators use different tokenizer.
        agent.add_argument('--load-candidate-vecs', type=bool, default=True)
        agent.add_argument('--load-candidate-retriever-vecs-path', type=str)
        agent.add_argument('--load-candidate-generator-vecs-path', type=str)
        agent.add_argument('--save-candidate-retriever-vecs-path', type=str, default=None)
        agent.add_argument('--save-candidate-generator-vecs-path', type=str, default=None)
        agent.add_argument('--save-generated-samples', type=bool, default=True,
                           help='Option to save generated responses from the model after inference.')
        agent.add_argument('--generation-result-path', type=str,
                           help='Where to save the generated responses.')
        agent.add_argument('--num-generated-responses', type=int, default=0,
                           help='How many responses are you willing to generate by inference.'
                                'Too big number will delay the training.')

        super().add_cmdline_args(parser, partial_opt=partial_opt)
        return agent

    def __init__(self, opt, shared=None):
        self.generated_samples = []
        self._load_corge_opt_and_dict(opt, shared)
        if shared:
            self.fixed_candidates = shared['fixed_candidates']
            self.fixed_candidate_retriever_vecs = shared['fixed_candidate_retriever_vecs']
            self.fixed_candidate_generator_vecs = shared['fixed_candidate_generator_vecs']

        super().__init__(opt, shared)
        self.local_text_logger: Dict[str, Any] = {}
        if not shared:
            self.model.prepare_faiss_index(self.opt)

    def save(self, path=None):
        """
        Save model parameters to path (or default to model_file arg).

        Please try to refrain from overriding this function, and instead override
        `state_dict(self)` for more specific saving.
        """
        super().save(path)
        # checkpoint saving during training
        path = self.opt.get('model_file', None) if path is None else path

        if path:
            states = self.state_dict()
            if states:  # anything found to save?
                if "checkpoint" not in path:
                    return
                temp_path = path + f"-step{self._number_training_updates}"
                atomic_save(states, temp_path)
                self.opt.save(temp_path + '.opt')
                logging.debug(f'Saving dictionary to {temp_path + ".dict"}')
                self.dict.save(temp_path + ".dict", sort=False)

    def _dummy_batch(self, batchsize, maxlen):
        """
        Create a dummy batch.

        This is used to preinitialize the cuda buffer, or otherwise force a
        null backward pass after an OOM.
        """
        batch = super()._dummy_batch(batchsize, maxlen)

        retriever_end_token = self.retriever_dict.end_token
        retriever_null_token = self.retriever_dict.null_token
        retriever_label_vec = (
            torch.LongTensor([self.retriever_dict.tok2ind[retriever_end_token],
                              self.retriever_dict.tok2ind[retriever_null_token]])
            .unsqueeze(0)
            .expand(batchsize, 2)
            .cuda()
        )

        batch.retriever_text_vec = batch.text_vec
        batch.retriever_label_vec = retriever_label_vec
        return batch

    def share(self):
        """
        Share fields from parent as well as useful objects in this class.
        """
        shared = super().share()
        shared['retriever_opt'] = self.retriever_opt
        shared['generator_opt'] = self.generator_opt
        shared['retriever_dict'] = self.retriever_dict
        shared['generator_dict'] = self.generator_dict
        shared['fixed_candidates'] = self.fixed_candidates
        shared['fixed_candidate_retriever_vecs'] = self.fixed_candidate_retriever_vecs
        shared['fixed_candidate_generator_vecs'] = self.fixed_candidate_generator_vecs
        return shared

    def _load_corge_opt_and_dict(self, opt, shared):
        """
        Load retriever and generator opt and dict from the pre-trained models.
        """
        if shared:
            self.retriever_opt = shared['retriever_opt']
            self.generator_opt = shared['generator_opt']
            self.retriever_dict = shared['retriever_dict']
            self.generator_dict = shared['generator_dict']
        else:
            self.retriever_opt = opt.load(opt["retriever_model_path"] + ".opt")
            self.generator_opt = opt.load(opt["generator_model_path"] + ".opt")
            self.generator_opt["n_positions"] = self.generator_opt["label_truncate"] + \
                self.generator_opt["text_truncate"]

            self.retriever_opt["dict_file"] = opt["retriever_model_path"] + ".dict"
            self.generator_opt["dict_file"] = opt["generator_model_path"] + ".dict"

            self.retriever_dict = DictionaryAgent(self.retriever_opt)
            self.generator_dict = DictionaryAgent(self.generator_opt)

    def vectorize(
        self,
        obs,
        history,
        add_start=True,
        add_end=True,
        text_truncate=None,
        label_truncate=None,
    ):
        obs = super().vectorize(
            obs=obs, history=history, add_start=add_start,
            add_end=add_end, text_truncate=text_truncate, label_truncate=label_truncate,
        )
        self._set_retriever_text_vec(obs, history, self.retriever_opt["text_truncate"])
        self._set_retriever_label_vec(obs, self.retriever_opt["label_truncate"])
        return obs

    def _set_retriever_text_vec(self, obs, history, truncate):
        if 'text' not in obs:
            return obs

        if 'retriever_text_vec' not in obs:
            history_string = history.get_history_str()
            obs['retriever_text_vec'] = self.retriever_dict.txt2vec(history_string)

        if obs.get('retriever_text_vec') is not None:
            truncate_left = not self.history_reversed
            truncated_vec = self._check_truncate(
                obs['retriever_text_vec'], truncate, truncate_left,
            )
            obs.force_set('retriever_text_vec', torch.LongTensor(truncated_vec))
            obs.force_set(
                'retriever_text_vec',
                self._retriever_add_start_end_tokens(obs['retriever_text_vec'], True, True)
            )
        return obs

    def _set_retriever_label_vec(self, obs, truncate):
        if 'labels' in obs:
            label_type = 'labels'
        elif 'eval_labels' in obs:
            label_type = 'eval_labels'
        else:
            label_type = None

        if label_type is None:
            return

        elif 'retriever_' + label_type + '_vec' in obs:
            # check truncation of pre-computed vector
            truncated_vec = self._check_truncate(obs['retriever_' + label_type + '_vec'],
                                                 truncate)
            obs.force_set('retriver_' + label_type + '_vec', torch.LongTensor(truncated_vec))
        else:
            # pick one label if there are multiple
            lbls = obs[label_type]
            label = lbls[0] if len(lbls) == 1 else self.random.choice(lbls)
            vec_label = self._retriever_vectorize_text(label, truncate, False)
            obs['retriever_' + label_type + '_vec'] = vec_label
            obs['retriever_' + label_type + '_choice'] = label

    def _retriever_add_start_end_tokens(self, vec, add_start=False, add_end=False):
        """
        Add start and end tokens to a list or tensor.
        """
        START_IDX = self.retriever_dict[self.retriever_dict.start_token]
        END_IDX = self.retriever_dict[self.retriever_dict.end_token]
        if isinstance(vec, torch.Tensor):
            if len(vec.shape) != 1:
                raise Exception('_add_start_end_tokens expects a 1D tensor')
            tensors = [vec]
            if add_start:
                tensors.insert(0, vec.new_tensor([START_IDX]))
            if add_end:
                tensors.append(vec.new_tensor([END_IDX]))
            return torch.cat(tensors, 0)
        if add_start:
            vec.insert(0, START_IDX)
        if add_end:
            vec.append(END_IDX)
        return vec

    def _retriever_vectorize_text(
        self, text, truncate=None, truncate_left=True,
    ):
        vec = self.retriever_dict.txt2vec(text)
        vec = self._retriever_add_start_end_tokens(vec, add_start=True, add_end=True)
        vec = self._check_truncate(vec, truncate, truncate_left)
        tensor = torch.LongTensor(vec)
        return tensor

    def batchify(self, obs_batch, sort=False):
        if len(obs_batch) == 0:
            return Batch(batchsize=0)

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]

        if len(valid_obs) == 0:
            return Batch(batchsize=0)

        valid_inds, exs = zip(*valid_obs)
        batch = super().batchify(obs_batch, sort=sort)

        # retriever text vec
        xs, x_lens = None, None
        if any(ex.get('retriever_text_vec') is not None for ex in exs):
            _xs = [ex.get('retriever_text_vec', self.EMPTY) for ex in exs]
            xs, x_lens = self._pad_tensor(_xs)
            if sort:
                sort = False  # now we won't sort on labels
                xs, x_lens, valid_inds, exs = argsort(
                    x_lens, xs, x_lens, valid_inds, exs, descending=True
                )

        # retriever label vec
        labels_avail = any('retriever_labels_vec' in ex for ex in exs)
        some_labels_avail = labels_avail or any('retriever_eval_labels_vec' in ex for ex in exs)

        ys, y_lens, labels = None, None, None
        if some_labels_avail:
            field = 'retriever_labels' if labels_avail else 'retriever_eval_labels'

            label_vecs = [ex.get(field + '_vec', self.EMPTY) for ex in exs]
            labels = [ex.get(field + '_choice') for ex in exs]
            y_lens = [y.shape[0] for y in label_vecs]

            ys, y_lens = self._pad_tensor(label_vecs)

            if sort and xs is None:
                ys, valid_inds, label_vecs, labels, y_lens = argsort(
                    y_lens, ys, valid_inds, label_vecs, labels, y_lens, descending=True
                )

        batch["retriever_text_vec"] = xs
        batch["retriever_text_lengths"] = x_lens
        batch["retriever_label_vec"] = ys
        batch["retriever_label_lengths"] = y_lens

        return batch

    def _set_fixed_candidates(self, opt):
        """
        Load candidates and fix them for exemplar-based generator.
        For efficient experiments, we allow to save the candidate vectors and load them before training.
        """
        cand_path = opt["fixed_candidates_path"]
        with PathManager.open(cand_path, 'r', encoding='utf-8') as f:
            cands = [line.strip() for line in f.readlines()]

        self.fixed_candidates = cands

        # loading candidate vecs for generator and retriever
        if (
            opt["load_candidate_vecs"] and
            Path(opt["load_candidate_retriever_vecs_path"]).exists() and
            Path(opt["load_candidate_generator_vecs_path"]).exists()
        ):
            logging.info(f"Load candidate vectors from {opt['load_candidate_retriever_vecs_path']} and "
                         f"{opt['load_candidate_retriever_vecs_path']}.")
            self.fixed_candidate_retriever_vecs = [
                torch.LongTensor(v) for v in np.load(opt["load_candidate_retriever_vecs_path"],
                                                     allow_pickle=True)
            ]
            self.fixed_candidate_generator_vecs = [
                torch.LongTensor(v) for v in np.load(opt["load_candidate_generator_vecs_path"],
                                                     allow_pickle=True)
            ]
        else:
            logging.info(
                f"Vectorizing fixed candidate set ({len(cands)})"
            )
            cand_vecs = []
            # store text vectors which is obtained from retriever dictionary.
            for cand in tqdm(cands):
                cand_vecs.append(self._retriever_vectorize_text(
                    cand,
                    truncate=self.retriever_opt["label_truncate"],
                    truncate_left=False,
                ))
            self.fixed_candidate_retriever_vecs = cand_vecs

            # store text vectors which is obtained from generator dictionary.
            text_vecs = []
            for cand in tqdm(cands):
                text_vecs.append(self._vectorize_text(
                    cand,
                    add_start=False,
                    add_end=False,
                    truncate=self.generator_opt["label_truncate"],
                    truncate_left=False,
                ))
            self.fixed_candidate_generator_vecs = text_vecs
            if opt["save_candidate_retriever_vecs_path"] is not None:
                vecs = [v.data.numpy() for v in self.fixed_candidate_retriever_vecs]
                np.save(opt["save_candidate_retriever_vecs_path"], vecs)
            if opt["save_candidate_generator_vecs_path"] is not None:
                vecs = [v.data.numpy() for v in self.fixed_candidate_generator_vecs]
                np.save(opt["save_candidate_generator_vecs_path"], vecs)

    def build_model(self, states=None):
        # build fixed candidates and feed them into RagModule for retrieval.
        self._set_fixed_candidates(self.opt)
        model = ExemplarBasedGeneratorModule(
            self.opt,
            self.retriever_opt,
            self.retriever_dict,
            self.generator_opt,
            self.generator_dict,
            self.fixed_candidates,
            self.fixed_candidate_retriever_vecs,
            self.fixed_candidate_generator_vecs,
        )
        return model

    def build_criterion(self):
        """
        Construct and return the loss function.
        """
        return eval(self.opt["criterion_type"])(null_idx=self.NULL_IDX, use_fp16=self.opt["fp16"])

    def eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_vec is None and batch.image is None:
            return
        self.model.eval()
        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss, model_output = self.compute_loss(batch, return_output=True, mode='eval')

        preds = None
        current_step = self._number_training_updates
        Path(self.opt["generation_result_path"]).mkdir(parents=True, exist_ok=True)
        if self.skip_generation:
            warn_once("--skip-generation true produces limited metrics")
        else:
            maxlen = self.label_truncate or 256
            beam_preds_scores, beams, list_chosen_candidates = self._generate(
                batch, self.beam_size, maxlen, evaluation=True,
            )
            preds, scores = zip(*beam_preds_scores)

            # bsz x beamsize
            beam_texts: List[List[Tuple[str, float]]] = []
            for beam in beams:
                beam_texts.append([])
                for tokens, score in beam.get_rescored_finished():
                    try:
                        beam_texts[-1].append((self._v2t(tokens), score.item()))
                    except KeyError:
                        logging.error("Decoding error: %s", tokens)
                        continue

            if self.opt["save_generated_samples"]:
                self.record_generation_results(batch, beam_texts, list_chosen_candidates,
                                               file_name=f"step{current_step}-generation-nogold.json")

        text = [self._v2t(p) for p in preds] if preds is not None else None
        if text is not None:
            self.local_text_logger["generated_text"] = text[0]

        return Output(text, None, token_losses=None)

    def record_generation_results(self, batch, beam_texts, list_chosen_candidates, file_name):
        # record the generation results on the file.
        Path(self.opt["generation_result_path"]).mkdir(parents=True, exist_ok=True)

        log_save_path = Path(self.opt["generation_result_path"]) / file_name
        observations = batch.observations
        for candidate, beam_text, observation in zip(list_chosen_candidates, beam_texts, observations):
            context = observation["full_text"]
            gold_response = observation["eval_labels"]
            result = {
                "context": context,
                "gold_response": gold_response,
                "candidate": candidate,
                "beam_text": beam_text[0][0],
                "score": beam_text[0][1],
            }
            self.generated_samples.append(result)

        if len(self.generated_samples) >= self.opt["num_generated_responses"]:
            with open(log_save_path, "w") as f:
                json.dump(self.generated_samples, f, indent=4)
            self.generated_samples = []
            self.skip_generation = True

    def report(self):
        # To report which exemplars are chosen during training.
        for k, v in self.local_text_logger.items():
            if isinstance(v, str):
                self.global_metrics._data[k] = TextMetric(v)
            elif isinstance(v, list) and all(isinstance(s, str) for s in v):
                for i, _v in enumerate(v):
                    self.global_metrics._data[f"{k}-{i}"] = TextMetric(_v)
        return super().report()

    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
        evaluation: bool = False,
        use_cand_encoder: bool = False,
    ):
        """
        Generate an output with beam search.
        Depending on the options, this may perform greedy/topk/nucleus generation.
        :param Batch batch:
            Batch structure with input and labels
        :param int beam_size:
            Size of each beam during the search
        :param int max_ts:
            the maximum length of the decoded sequence
        :param prefix_tokens:
            if given, a tensor of tokens that must begin the decoded sequence.
        :return:
            tuple (beam_pred_scores, beams)
            - beam_preds_scores: list of (prediction, score) pairs for each sample in
              Batch
            - beams :list of Beam instances defined in Beam class, can be used for any
              following postprocessing, e.g. dot logging.
        """
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            # TODO make DDP works!
            raise NotImplementedError

        topk = 1
        retrieved_topk_candidates, candidate_scores, list_chosen_candidates, encoder_states = \
            self.model.prepare_generator_input(batch, topk, use_cand_encoder=False)

        if batch.text_vec is not None:
            dev = batch.text_vec.device
        else:
            assert batch.label_vec is not None, "need label_vec for _generate"
            dev = batch.label_vec.device

        bsz = (
            len(batch.text_lengths)
            if batch.text_lengths is not None
            else len(batch.image)  # type: ignore
        )
        if batch.text_vec is not None:
            batchsize = batch.text_vec.size(0)
            beams = [
                self._treesearch_factory(dev)
                .set_context(self._get_context(batch, batch_idx))
                .set_block_list(self.beam_block_list)
                for batch_idx in range(batchsize)
            ]
        else:
            beams = [self._treesearch_factory(dev) for _ in range(bsz)]

        # repeat encoder outputs and decoder inputs
        decoder_input = self._get_initial_decoder_input(bsz, beam_size, dev)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = self.model.generator.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            score, incr_state = self.model.generator.decoder(decoder_input, encoder_states, incr_state)
            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            score = self.model.generator.output(score)
            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            if self.temperature != 1.0:
                score.div_(self.temperature)
            # force to fp32 to avoid overflow issues during search calculations
            score = torch.nn.functional.log_softmax(score, dim=-1, dtype=torch.float32)  # type: ignore
            if prefix_tokens is not None and _ts < prefix_tokens.size(1):
                # generate prefix_tokens for every timestep that they exist
                # achieve by setting score of all other tokens to be -inf
                prefix_toks = prefix_tokens[:, _ts].unsqueeze(-1).repeat(1, beam_size)
                prefix_score = score.gather(-1, prefix_toks.unsqueeze(-1))
                prefix_mask = prefix_toks.ne(self.NULL_IDX)
                score[prefix_mask] = neginf(score.dtype)
                score[prefix_mask] = score[prefix_mask].scatter_(
                    -1,
                    prefix_toks[prefix_mask].unsqueeze(-1),
                    prefix_score[prefix_mask],
                )
            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i])
            incr_state_inds = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            incr_state = self.model.generator.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            decoder_input = self._get_next_decoder_input(
                decoder_input, selection, incr_state_inds
            )

        # get all finalized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]

        if hasattr(self, '_rerank_beams'):
            n_best_beam_preds_scores = self._rerank_beams(  # type: ignore
                batch, n_best_beam_preds_scores
            )

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [n_best_list[0] for n_best_list in n_best_beam_preds_scores]

        if evaluation:
            return beam_preds_scores, beams, list_chosen_candidates
        else:
            return beam_preds_scores, beams

    def compute_loss(self, batch, return_output=False, mode="train"):
        """
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')

        model_output = self.model(batch,
                                  topk=self.opt["corge_topk_cands"],
                                  use_cand_encoder=True)
        scores, candidate_scores, preds, _, list_chosen_candidates = model_output

        bsz, num_cands = candidate_scores.shape
        labels_repeat = torch.unsqueeze(batch.label_vec, dim=1).repeat(
            1, num_cands, 1)
        final_loss, log_likelihood_loss = self.criterion(scores,
                                                         candidate_scores,
                                                         labels_repeat)
        sum_log_likelihood_loss = log_likelihood_loss.sum(-1)  # [bsz]

        notnull = labels_repeat.ne(self.NULL_IDX)  # [bsz, num_cands, seq_len]
        correct = ((preds.contiguous().view(bsz, num_cands, -1)
                    == labels_repeat) * notnull.float())  # [bsz, num_cands, seq_len]
        sum_correct = correct.contiguous().view(bsz, -1).sum(-1)  # [bsz]
        sum_target_tokens = notnull.long().contiguous().view(bsz, -1).sum(-1)  # [bsz]

        self.record_local_metric('nll_loss', AverageMetric.many(sum_log_likelihood_loss,
                                                                sum_target_tokens))
        self.record_local_metric('ppl', PPLMetric.many(sum_log_likelihood_loss,
                                                       sum_target_tokens))
        self.record_local_metric('token_acc', AverageMetric.many(sum_correct, sum_target_tokens))
        self.record_local_metric('loss', AverageMetric.many(final_loss, sum_target_tokens))
        self.record_local_metric('avg_cand_scores', AverageMetric.many(candidate_scores.mean(-1),
                                                                       torch.ones(bsz).long()))

        softmax_scores = torch.softmax(candidate_scores, dim=-1)
        self.record_local_metric('avg_cand_probs', AverageMetric.many(softmax_scores.mean(-1),
                                                                      torch.ones(bsz).long()))

        # store training samples and retrieved candidates for transparency
        if batch.observations is not None:
            self.local_text_logger["context"] = batch.observations[0]["full_text"]
            self.local_text_logger["gold_response"] = batch.observations[0]["labels"][0] if mode == "train" \
                else batch.observations[0]["eval_labels"][0]
            self.local_text_logger["exemplar"] = list_chosen_candidates[0]

        if return_output:
            return (final_loss, model_output)
        else:
            return final_loss.mean()
