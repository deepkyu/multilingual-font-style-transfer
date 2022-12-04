from pathlib import Path
import shutil


def save_files(path_save_, savefiles):
    path_save = Path(path_save_)
    path_save.mkdir(exist_ok=True)

    for savefile in savefiles:
        parents_dir = Path(savefile).parents
        if len(parents_dir) >= 1:
            for parent_dir in list(parents_dir)[::-1]:
                target_dir = path_save / parent_dir
                target_dir.mkdir(exist_ok=True)
        try:
            shutil.copy2(savefile, str(path_save / savefile))
        except Exception as e:
            # skip the file
            print(f'{e} occured while saving {savefile}')

    return  # success


if __name__ == "__main__":
    import glob
    savefiles = glob.glob('config/*.yaml')
    savefiles += glob.glob('config/**/*.yaml')
    save_files(".temp", savefiles)
