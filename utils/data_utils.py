import pandas as pd

def sample_data(df, strategy="undersample", oversample_factor=2, undersample_ratio=0.7, balanced_neg_ratio=0.5, random_state=42):
    biased = df[df.label == 1]
    unbiased = df[df.label == 0]

    if strategy == "undersample":
        # Undersample the unbiased (majority) class to match a specified ratio
        unbiased_sampled = unbiased.sample(frac=undersample_ratio, random_state=random_state)
        return pd.concat([biased, unbiased_sampled])

    elif strategy == "oversample":
        # Duplicate the biased (minority) class oversample_factor times
        return pd.concat([biased] * oversample_factor + [unbiased])

    elif strategy == "balanced":
        # Target 50% biased, 50% unbiased in the final dataset (or as specified by balanced_neg_ratio)
        target_total = len(df)  # preserve the original size
        unbiased_target = int(target_total * balanced_neg_ratio)  # target number of unbiased samples  
        biased_target = target_total - unbiased_target  # target number of biased samples 

        # Sample unbiased samples to match the target
        neg_sampled = unbiased.sample(n=unbiased_target, random_state=random_state)

        # Compute the number of times to duplicate all biased samples
        repeats = biased_target // len(biased)
        # Compute the remainder of biased samples to sample to reach target
        remainder = biased_target % len(biased)

        # Sample biased samples to match the target based on the number of repeats and remainder
        biased_repeated = pd.concat([biased] * repeats + [biased.sample(n=remainder, random_state=random_state)])
        return pd.concat([biased_repeated, neg_sampled])

    elif strategy == "normal":
        return df

    else:
        raise ValueError("Unsupported strategy.")
