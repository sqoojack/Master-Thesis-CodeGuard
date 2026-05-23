# shadowcode.py
import torch
import itertools
from tqdm import tqdm
from config import Config
from utils import calc_fre_loss
from objectives import ObjectiveFactory


class ShadowCodeAttacker:
    def __init__(self, model, tokenizer, config: Config, lang, attack_tokenizer=None):
        """
        Args:
            model:
                Surrogate model used for ShadowCode optimization.

            tokenizer:
                Tokenizer paired with the surrogate model.
                This tokenizer MUST match the surrogate model because it is used
                for model forward, loss computation, gradient computation, and decoding.

            config:
                ShadowCode Config.

            lang:
                Programming language. One of: python, c, cpp, java.

            attack_tokenizer:
                Optional tokenizer used only to define the attack token search space.
                If None, fallback to the surrogate tokenizer.

        Important:
            Do NOT directly feed attack_tokenizer token ids into the surrogate model
            unless attack_tokenizer == tokenizer. Token ids are model-specific.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.attack_tokenizer = attack_tokenizer if attack_tokenizer is not None else tokenizer
        self.config = config
        self.lang = lang

        self.embedding_layer = model.get_input_embeddings()

        # Optional restricted search space for tokenizer-transfer experiments.
        # These ids are always surrogate-model token ids.
        self.allowed_perturbation_ids = self._build_allowed_perturbation_ids()

    def _same_tokenizer(self):
        """
        Return True if attack_tokenizer and surrogate tokenizer should be treated
        as the same tokenizer.
        """
        if self.attack_tokenizer is self.tokenizer:
            return True

        attack_name = getattr(self.attack_tokenizer, "name_or_path", None)
        model_name = getattr(self.tokenizer, "name_or_path", None)

        return attack_name is not None and model_name is not None and attack_name == model_name

    def _build_allowed_perturbation_ids(self):
        """
        Build a restricted perturbation search space.

        For normal ShadowCode:
            attack_tokenizer == tokenizer
            => return None
            => perturbation search uses the full surrogate model vocabulary.

        For tokenizer-transfer:
            attack_tokenizer != tokenizer
            => use attack_tokenizer vocab to generate text pieces
            => re-tokenize each text piece with surrogate tokenizer
            => keep only pieces that map to exactly one surrogate token id

        This preserves the existing fixed-length perturbation optimization.
        """
        if self._same_tokenizer():
            return None

        print("[*] Building tokenizer-transfer perturbation search space...")

        allowed_ids = set()
        attack_vocab = self.attack_tokenizer.get_vocab()

        for attack_token_id in attack_vocab.values():
            try:
                token_text = self.attack_tokenizer.decode(
                    [attack_token_id],
                    skip_special_tokens=True
                )
            except Exception:
                continue

            if token_text is None or token_text == "":
                continue

            try:
                model_ids = self.tokenizer(
                    token_text,
                    return_tensors="pt",
                    add_special_tokens=False
                ).input_ids[0]
            except Exception:
                continue

            # Keep only 1-to-1 token mapping.
            # This is necessary because current ShadowCode optimizes a fixed number
            # of token slots in surrogate embedding space.
            if model_ids.numel() == 1:
                model_id = int(model_ids.item())
                if 0 <= model_id < self.model.config.vocab_size:
                    allowed_ids.add(model_id)

        if len(allowed_ids) == 0:
            print("[!] Warning: tokenizer-transfer perturbation search space is empty.")
            print("[!] Falling back to full surrogate model vocabulary.")
            return None

        allowed_ids = torch.tensor(
            sorted(allowed_ids),
            dtype=torch.long,
            device=self.config.DEVICE
        )

        print(
            f"[*] Allowed perturbation token ids: "
            f"{len(allowed_ids)} / {self.model.config.vocab_size}"
        )

        return allowed_ids

    def _build_keyword_candidate_pool(self, simulation):
        """
        Build keyword candidate pool.

        Normal setting:
            Use surrogate tokenizer directly.

        Tokenizer-transfer setting:
            1. Extract candidate tokens under attack_tokenizer.
            2. Decode each attack-token id to text.
            3. Re-tokenize text with surrogate tokenizer.
            4. Keep only candidates mapping to exactly one surrogate token.
        """
        if self._same_tokenizer():
            return ObjectiveFactory.extract_candidate_tokens(
                simulation,
                self.tokenizer
            ).to(self.config.DEVICE)

        print("[*] Building tokenizer-transfer keyword candidate pool...")

        attack_pool = ObjectiveFactory.extract_candidate_tokens(
            simulation,
            self.attack_tokenizer
        )

        model_token_ids = []

        for attack_token_id in attack_pool.tolist():
            try:
                token_text = self.attack_tokenizer.decode(
                    [attack_token_id],
                    skip_special_tokens=True
                )
            except Exception:
                continue

            if token_text is None or token_text == "":
                continue

            try:
                encoded = self.tokenizer(
                    token_text,
                    return_tensors="pt",
                    add_special_tokens=False
                ).input_ids[0]
            except Exception:
                continue

            # Keep only one-token mappings.
            if encoded.numel() == 1:
                model_token_id = int(encoded.item())
                if 0 <= model_token_id < self.model.config.vocab_size:
                    model_token_ids.append(model_token_id)

        model_token_ids = sorted(set(model_token_ids))

        if len(model_token_ids) == 0:
            print("[!] Warning: keyword candidate pool is empty after tokenizer remapping.")
            print("[!] Falling back to surrogate tokenizer candidate pool.")
            return ObjectiveFactory.extract_candidate_tokens(
                simulation,
                self.tokenizer
            ).to(self.config.DEVICE)

        candidate_pool = torch.tensor(
            model_token_ids,
            dtype=torch.long,
            device=self.config.DEVICE
        )

        print(f"[*] Keyword candidate pool after remapping: {len(candidate_pool)} tokens")
        return candidate_pool

    def _init_perturbation_ids(self):
        """
        Initialize perturbation ids.

        If allowed_perturbation_ids is None:
            sample from full surrogate model vocab.

        Else:
            sample only from allowed surrogate-token ids derived from attack_tokenizer.
        """
        if self.allowed_perturbation_ids is None:
            return torch.randint(
                0,
                self.model.config.vocab_size,
                (1, self.config.PERTURBATION_LEN),
                device=self.config.DEVICE
            )

        rand_idx = torch.randint(
            0,
            len(self.allowed_perturbation_ids),
            (1, self.config.PERTURBATION_LEN),
            device=self.config.DEVICE
        )

        return self.allowed_perturbation_ids[rand_idx]

    def run_alternating_attack(self, simulation):
        print(f"Starting Robust Alternating Attack on Target: {simulation.target_code.strip()}")

        # Suffix = trigger position.
        suffix_str = f"\n{simulation.position_code}"
        suffix_ids = self.tokenizer(
            suffix_str,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.config.DEVICE)

        target_ids = self.tokenizer(
            simulation.target_code,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.config.DEVICE)

        candidate_pool = self._build_keyword_candidate_pool(simulation)
        print(f"Keyword Candidate Pool Size: {len(candidate_pool)} tokens")

        if len(candidate_pool) < self.config.KEYWORD_LEN:
            raise ValueError(
                f"candidate_pool size ({len(candidate_pool)}) is smaller than "
                f"KEYWORD_LEN ({self.config.KEYWORD_LEN})."
            )

        torch.manual_seed(self.config.SEED)

        perturbation_ids = self._init_perturbation_ids()

        keyword_ids = candidate_pool[
            torch.randperm(len(candidate_pool), device=self.config.DEVICE)[:self.config.KEYWORD_LEN]
        ].unsqueeze(0)

        best_loss = float("inf")
        best_keyword_ids = keyword_ids.clone()
        best_perturbation_ids = perturbation_ids.clone()

        torch.cuda.empty_cache()

        for epoch in range(self.config.ALTERNATING_EPOCHS):
            print(f"\n=== Epoch {epoch + 1}/{self.config.ALTERNATING_EPOCHS} ===")

            # 1. Optimize perturbation by robust gradient search.
            print(">> Optimizing Perturbation (Robust Gradient)...")
            perturbation_ids, p_loss = self._optimize_perturbation_robust(
                simulation,
                keyword_ids,
                perturbation_ids,
                suffix_ids,
                target_ids
            )
            print(f"   Best Perturbation Loss: {p_loss:.4f}")
            torch.cuda.empty_cache()

            # 2. Optimize keyword by grid search.
            print(">> Optimizing Keyword (Grid Search)...")
            keyword_ids, k_loss = self._optimize_keyword_grid(
                simulation,
                candidate_pool,
                perturbation_ids,
                suffix_ids,
                target_ids
            )
            print(f"   Best Keyword Loss: {k_loss:.4f}")
            torch.cuda.empty_cache()

            if k_loss < best_loss:
                best_loss = k_loss
                best_keyword_ids = keyword_ids.clone()
                best_perturbation_ids = perturbation_ids.clone()
                print(f"   [New Best] Loss: {best_loss:.4f}")

        final_keyword = self.tokenizer.decode(
            best_keyword_ids[0],
            skip_special_tokens=True
        )
        final_perturbation = self.tokenizer.decode(
            best_perturbation_ids[0],
            skip_special_tokens=True
        )

        return final_keyword, final_perturbation

    def _get_comment_prefix_ids(self):
        """
        Get language-specific comment prefix.

        C/C++/Java:
            //

        Python:
            #
        """
        marker = "// " if self.lang in ["c", "cpp", "java"] else "# "

        return self.tokenizer(
            marker,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.config.DEVICE)

    def _optimize_perturbation_robust(self, simulation, keyword, perturbation, suffix, target):
        current_perturbation = perturbation.clone()
        embed_weights = self.embedding_layer.weight
        best_overall_loss = float("inf")

        num_grad_samples = 5

        cond_ids = self.tokenizer(
            simulation.conditional_code,
            return_tensors="pt",
            add_special_tokens=True
        ).input_ids.to(self.config.DEVICE)

        # Correct language-specific comment prefix.
        comment_prefix_ids = self._get_comment_prefix_ids()

        pbar = tqdm(
            range(self.config.PERTURB_ITERATIONS),
            desc="Robust Iter",
            leave=False
        )

        for _ in pbar:
            for token_idx in range(self.config.PERTURBATION_LEN):
                # Step 1: Compute robust average gradient.
                total_grad = 0
                part1 = torch.cat([cond_ids, comment_prefix_ids, keyword], dim=1)

                for _ in range(num_grad_samples):
                    noise_str = simulation.get_noise_code()

                    noise_ids = self.tokenizer(
                        noise_str,
                        return_tensors="pt",
                        add_special_tokens=False
                    ).input_ids.to(self.config.DEVICE)

                    if noise_ids.shape[1] > 256:
                        noise_ids = noise_ids[:, :256]

                    part3 = torch.cat([noise_ids, suffix], dim=1)

                    grad, _ = self.token_gradients_with_suffix(
                        part1,
                        current_perturbation,
                        part3,
                        target
                    )

                    total_grad += grad

                    del noise_ids, part3, grad

                avg_grad = total_grad / num_grad_samples
                del total_grad

                # Step 2: Select top-k candidate tokens for the current slot.
                token_grad = avg_grad[:, token_idx, :]
                candidate_scores = -torch.matmul(token_grad, embed_weights.t())

                if self.allowed_perturbation_ids is not None:
                    restricted_scores = candidate_scores[:, self.allowed_perturbation_ids]
                    real_top_k = min(self.config.TOP_K, restricted_scores.shape[1])

                    top_pos = torch.topk(
                        restricted_scores,
                        real_top_k,
                        dim=1
                    ).indices[0]

                    top_k_indices = self.allowed_perturbation_ids[top_pos]
                else:
                    real_top_k = min(self.config.TOP_K, candidate_scores.shape[1])

                    top_k_indices = torch.topk(
                        candidate_scores,
                        real_top_k,
                        dim=1
                    ).indices[0]

                candidate_batch = current_perturbation.repeat(real_top_k, 1)
                candidate_batch[:, token_idx] = top_k_indices

                eval_noise_str = simulation.get_noise_code()

                eval_noise_ids = self.tokenizer(
                    eval_noise_str,
                    return_tensors="pt",
                    add_special_tokens=False
                ).input_ids.to(self.config.DEVICE)

                if eval_noise_ids.shape[1] > 256:
                    eval_noise_ids = eval_noise_ids[:, :256]

                eval_part3 = torch.cat([eval_noise_ids, suffix], dim=1)

                current_loss = self.evaluate_batch_variable_component(
                    part1,
                    current_perturbation,
                    eval_part3,
                    eval_part3,
                    target,
                    component="perturbation",
                    show_progress=False,
                    use_fre=True
                )[0].item()

                candidate_losses = self.evaluate_batch_variable_component(
                    part1,
                    candidate_batch,
                    eval_part3,
                    eval_part3,
                    target,
                    component="perturbation",
                    show_progress=False,
                    use_fre=True
                )

                min_loss_idx = torch.argmin(candidate_losses)
                min_loss = candidate_losses[min_loss_idx].item()

                if min_loss < current_loss:
                    current_perturbation = candidate_batch[min_loss_idx].unsqueeze(0)

                    if min_loss < best_overall_loss:
                        best_overall_loss = min_loss

                del avg_grad
                del candidate_batch
                del candidate_losses
                del eval_noise_ids
                del eval_part3

            pbar.set_postfix({"Best Loss": f"{best_overall_loss:.4f}"})

        return current_perturbation, best_overall_loss

    def _optimize_keyword_grid(self, simulation, candidate_pool, perturbation, suffix, target):
        pool_list = candidate_pool.tolist()
        combinations = list(itertools.product(pool_list, repeat=self.config.KEYWORD_LEN))

        candidate_keywords = torch.tensor(
            combinations,
            dtype=torch.long,
            device=self.config.DEVICE
        )

        cond_ids = self.tokenizer(
            simulation.conditional_code,
            return_tensors="pt",
            add_special_tokens=True
        ).input_ids.to(self.config.DEVICE)

        # Correct language-specific comment prefix.
        comment_prefix_ids = self._get_comment_prefix_ids()

        noise_str = simulation.get_noise_code()

        noise_ids = self.tokenizer(
            noise_str,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.config.DEVICE)

        if noise_ids.shape[1] > 256:
            noise_ids = noise_ids[:, :256]

        # [Cond] + [// ] or [# ]
        part1 = torch.cat([cond_ids, comment_prefix_ids], dim=1)

        # [Perturbation]
        part3 = perturbation

        # [Noise] + [Trigger]
        part4 = torch.cat([noise_ids, suffix], dim=1)

        losses = self.evaluate_batch_variable_component(
            part1,
            candidate_keywords,
            part3,
            part4,
            target,
            component="keyword",
            use_fre=True
        )

        best_idx = torch.argmin(losses)
        best_loss = losses[best_idx].item()
        best_keyword = candidate_keywords[best_idx].unsqueeze(0)

        del candidate_keywords
        del noise_ids
        del part4
        del losses

        return best_keyword, best_loss

    def evaluate_batch_variable_component(
        self,
        part1,
        variable_part,
        part3,
        part4,
        target,
        component="perturbation",
        show_progress=True,
        use_fre=False
    ):
        batch_size = self.config.EVAL_BATCH_SIZE
        num_candidates = variable_part.shape[0]
        all_losses = []

        iterator = range(0, num_candidates, batch_size)

        if show_progress and num_candidates > batch_size:
            iterator = tqdm(
                iterator,
                desc=f"Eval {component}",
                leave=False
            )

        with torch.no_grad():
            for i in iterator:
                var_batch = variable_part[i: i + batch_size]
                curr_bs = var_batch.shape[0]

                p1 = part1.expand(curr_bs, -1)
                p3 = part3.expand(curr_bs, -1)
                tgt = target.expand(curr_bs, -1)

                if component == "keyword":
                    p4 = part4.expand(curr_bs, -1)
                    full_input = torch.cat([p1, var_batch, p3, p4, tgt], dim=1)
                else:
                    full_input = torch.cat([p1, var_batch, p3, tgt], dim=1)

                # MAX_LENGTH truncation protection.
                if full_input.shape[1] > self.config.MAX_LENGTH:
                    full_input = full_input[:, -self.config.MAX_LENGTH:]

                outputs = self.model(input_ids=full_input)
                logits = outputs.logits

                tgt_len = target.shape[1]
                target_logits = logits[:, -tgt_len - 1: -1, :]

                if use_fre:
                    shift_logits = target_logits
                    shift_labels = tgt

                    loss_per_token = torch.nn.functional.cross_entropy(
                        shift_logits.transpose(1, 2),
                        shift_labels,
                        reduction="none"
                    )

                    l_origin = loss_per_token.mean(dim=1)

                    l_enhanced = loss_per_token[
                        :,
                        :min(self.config.FRE_H, loss_per_token.size(1))
                    ].mean(dim=1)

                    losses = (
                        l_origin + self.config.FRE_R * l_enhanced
                    ) / (self.config.FRE_R + 1)

                else:
                    losses = torch.nn.functional.cross_entropy(
                        target_logits.reshape(-1, target_logits.size(-1)),
                        tgt.reshape(-1),
                        reduction="none"
                    ).view(curr_bs, -1).mean(dim=1)

                all_losses.append(losses.cpu())

                del outputs
                del logits
                del target_logits
                del full_input

        return torch.cat(all_losses).to(self.config.DEVICE)

    def token_gradients_with_suffix(self, context_before, perturbation, suffix, target):
        full_input_ids = torch.cat(
            [context_before, perturbation, suffix, target],
            dim=1
        )

        if full_input_ids.shape[1] > self.config.MAX_LENGTH:
            full_input_ids = full_input_ids[:, -self.config.MAX_LENGTH:]

        inputs_embeds = self.embedding_layer(full_input_ids).detach()
        inputs_embeds.requires_grad = True

        # Because full_input_ids may be truncated, recompute perturbation offsets.
        p_len = perturbation.shape[1]
        p_end = full_input_ids.shape[1] - suffix.shape[1] - target.shape[1]
        p_start = p_end - p_len

        outputs = self.model(inputs_embeds=inputs_embeds)
        logits = outputs.logits

        tgt_len = target.shape[1]
        target_logits = logits[:, -tgt_len - 1: -1, :]

        loss = calc_fre_loss(target_logits, target, self.config)

        self.model.zero_grad()
        loss.backward()

        grad = inputs_embeds.grad[:, p_start:p_end, :]

        return grad, loss.item()