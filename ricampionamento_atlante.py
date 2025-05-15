import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def resample_nifti(input_path, output_path, target_voxel_size, order=0):
    """
    Ricampiona un'immagine NIfTI alla dimensione voxel desiderata.

    Parameters:
    - input_path: percorso file NIfTI originale
    - output_path: percorso file NIfTI di output ricampionato
    - target_voxel_size: tuple/list (sx, sy, sz) in mm, dimensione voxel desiderata
    - order: interpolazione (0=nearest neighbor, 1=bilinear, 3=cubic). Per atlanti categoriali usare 0.

    """

    # Carica l'immagine NIfTI
    img = nib.load(input_path)
    data = img.get_fdata()
    header = img.header.copy()
    affine = img.affine.copy()

    # Dimensioni voxel originali
    original_voxel_size = header.get_zooms()[:3]

    # Calcola i fattori di zoom per ciascuna dimensione
    zoom_factors = np.array(original_voxel_size) / np.array(target_voxel_size)
    print(f"Fattori di zoom: {zoom_factors}")

    # Ricampionamento dati (attenzione: potrebbe consumare memoria)
    data_resampled = zoom(data, zoom=zoom_factors, order=order)

    # Aggiorna l'header con la nuova dimensione voxel
    new_header = header.copy()
    new_header.set_zooms(target_voxel_size)

    # Calcola nuova affine con voxel size aggiornata mantenendo rotazioni
    new_affine = affine.copy()
    scaling = np.diag(target_voxel_size + (1,))
    new_affine[:3, :3] = affine[:3, :3] / np.diag(affine[:3, :3])[:, None] * target_voxel_size

    # Salva la nuova immagine NIfTI ricampionata
    new_img = nib.Nifti1Image(data_resampled, affine=new_affine, header=new_header)
    nib.save(new_img, output_path)
    print(f"Atlante ricampionato salvato in: {output_path}")

if __name__ == "__main__":
    input_atlas = "C:\\Users\\daria\\OneDrive\\Desktop\\ESAME\\BN_Atlas_246_2mm.nii.gz"        # Percorso atlante originale
    output_atlas = "C:\\Users\\daria\\OneDrive\\Desktop\\atlanteNUOVO_ricampionato_1_5.nii.gz" # Percorso atlante ricampionato
    target_voxel = (1.5, 1.5, 1.5)                   # Dimensioni voxel desiderate (mm)

    # Usa order=0 per dati categoriali (es. atlanti etichettati)
    resample_nifti(input_atlas, output_atlas, target_voxel, order=0)
