"""
FundusDataset e divisão de dados por paciente.

A divisão é feita no nível do paciente (não da imagem) para evitar data
leakage: imagens do mesmo paciente nunca aparecem em splits diferentes
simultaneamente. Isso é essencial em datasets longitudinais como o HYAMD,
onde um paciente pode ter múltiplas imagens adquiridas em visitas distintas.
(Seção III-D do relatório)
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from .preprocessing import preprocess_fundus


def load_and_binarize(labels_csv: str, images_dir: str) -> pd.DataFrame:
    """
    Carrega labels.csv e aplica a binarização do artigo de referência:
      AMD = 2        → label 1  (intermediário-tardio: NVAMD, GA, drusas grandes)
      AMD ∈ {0, 1}   → label 0  (controle + early AMD, agrupados como non-AMD)

    Retorna o DataFrame apenas com imagens cujos arquivos existem em images_dir.
    """
    df = pd.read_csv(labels_csv)
    df['label'] = (df['AMD'] == 2).astype(int)

    images_dir = Path(images_dir)
    # Files are named e.g. "000917597_L_1_.png" (trailing underscore before ext)
    # but image_id in the CSV is "000917597_L_1" — strip the trailing underscore.
    id_to_path = {f.stem.rstrip('_'): f for f in images_dir.iterdir() if f.is_file()}
    df['path'] = df['image_id'].map(id_to_path)

    n_missing = df['path'].isna().sum()
    if n_missing > 0:
        print(f'Aviso: {n_missing} imagens não encontradas e removidas.')
    return df[df['path'].notna()].copy()


def make_splits(df: pd.DataFrame, seed: int = 42):
    """
    Divisão em três partes por patient_id: ~64% treino / ~16% val / ~20% teste.

    Usa GroupShuffleSplit duas vezes para garantir que pacientes não se
    cruzem entre splits. Inclui assertions para confirmar ausência de leakage.

    Retorna (train_df, val_df, test_df).
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_val_idx, test_idx = next(gss.split(df, groups=df['patient_id']))
    train_val_df = df.iloc[train_val_idx]
    test_df      = df.iloc[test_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df['patient_id']))
    train_df = train_val_df.iloc[train_idx].copy()
    val_df   = train_val_df.iloc[val_idx].copy()

    assert len(set(train_df['patient_id']) & set(val_df['patient_id']))  == 0
    assert len(set(train_df['patient_id']) & set(test_df['patient_id'])) == 0
    assert len(set(val_df['patient_id'])   & set(test_df['patient_id'])) == 0

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class FundusDataset(Dataset):
    """Dataset PyTorch para imagens de fundo de olho do HYAMD.

    Aplica o pipeline de pré-processamento (crop + pad + resize) antes
    das transformações de augmentação/normalização.
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None, target_size: int = 224):
        self.df          = dataframe.reset_index(drop=True)
        self.transform   = transform
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row['path']).convert('RGB')
        img   = preprocess_fundus(img, target_size=self.target_size)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row['label'], dtype=torch.float32)
        return img, label
