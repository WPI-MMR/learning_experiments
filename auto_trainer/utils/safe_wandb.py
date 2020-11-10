def safe_import_wandb() -> bool:
  try:
    import wandb
    return wandb
  except ImportError:
    return False