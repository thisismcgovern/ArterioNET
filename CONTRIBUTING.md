# Contributing to ArterioNet

We welcome contributions! Here's how to get started.

## Setup

```bash
git clone https://github.com/mcgovern-twumasi/arterionet.git
cd arterionet
pip install -e ".[dev,mlx]"
```

## Development Workflow

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make changes and test
3. Run tests: `pytest tests/`
4. Format code: `black arterionet/`
5. Commit: `git commit -m "Add feature"`
6. Push and create a PR

## Code Standards

- PEP 8 style (Black formatter)
- Type hints for all functions
- Docstrings for all modules/classes
- Unit tests for new functionality

## Testing

```bash
pytest tests/ -v --cov=arterionet
```

## Reporting Issues

Use GitHub Issues with:
- Clear title
- Reproducible example
- Expected vs actual behavior
- Environment (OS, Python, CUDA/MPS version)

## Citation

When referring to ArterioNet in papers/projects, please cite:

```bibtex
@software{arterionet2024,
  title={ArterioNet: Cuffless ABP Reconstruction from ECG and PPG},
  author={Owusu-Bekoe, McGovern Twumasi and Apedo, Emefa Abena},
  year={2024},
  url={https://github.com/mcgovern-twumasi/arterionet}
}
```
