import argparse

from evo import Evo, generate


def main():

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description='Generate sequences using the Evo model.')

    parser.add_argument('--model_path', type=str, default='evo-1-131k-base', help='Evo model name')
    parser.add_argument('--dna', type=str, default='ACGT', help='Prompt for generation')
    parser.add_argument('--n_samples', type=str, default="3", help='Number of sequences to sample at once')
    parser.add_argument('--n_tokens', type=str, default="100", help='Number of tokens to generate')
    parser.add_argument('--temperature', type=str, default="1.0", help='Temperature during sampling')
    parser.add_argument('--top_k', type=str, default="4", help='Top K during sampling')
    parser.add_argument('--top_p', type=str, default="1.", help='Top P during sampling')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for generation')
    parser.add_argument('--verbose', type=str, default="1", help='Verbosity level')

    args = parser.parse_args()

    # Load model.

    evo_model = Evo(args.model_path)
    model, tokenizer = evo_model.model, evo_model.tokenizer

    model.to(args.device)
    model.eval()

    # Sample sequences.
    
    print('Generated sequences:')
    output_seqs, output_scores = generate(
        [ args.dna ] * int(args.n_samples),
        model,
        tokenizer,
        n_tokens=int(args.n_tokens),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        cached_generation=True,
        batched=True,
        prepend_bos=False,
        device=args.device,
        verbose=int(args.verbose),
    )


if __name__ == '__main__':
    main()