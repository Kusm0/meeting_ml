"""
Скрипт для завантаження датасету MeetingBank-transcript з Hugging Face.
"""
from datasets import load_dataset
from pathlib import Path
import logging

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_meetingbank_dataset(
    dataset_name: str = "lytang/MeetingBank-transcript",
    cache_dir: str = None,
    save_to_disk: bool = False,
    save_path: str = None
):
    """
    Завантажує датасет MeetingBank-transcript з Hugging Face.
    
    Args:
        dataset_name: Назва датасету на Hugging Face
        cache_dir: Директорія для кешування (за замовчуванням ~/.cache/huggingface)
        save_to_disk: Чи зберігати датасет на диск після завантаження
        save_path: Шлях для збереження датасету (якщо save_to_disk=True)
    
    Returns:
        DatasetDict або Dataset об'єкт з датасетом
    """
    try:
        logger.info(f"Початок завантаження датасету: {dataset_name}")
        
        # Завантаження датасету
        ds = load_dataset(
            dataset_name,
            cache_dir=cache_dir
        )
        
        logger.info("Датасет успішно завантажено!")
        logger.info(f"Тип датасету: {type(ds)}")
        
        # Виведення інформації про датасет
        if hasattr(ds, 'keys'):
            logger.info(f"Доступні розділи: {list(ds.keys())}")
            for split_name, split_data in ds.items():
                logger.info(f"  {split_name}: {len(split_data)} записів")
                if len(split_data) > 0:
                    logger.info(f"    Приклад ключів: {list(split_data[0].keys())}")
        else:
            logger.info(f"Кількість записів: {len(ds)}")
            if len(ds) > 0:
                logger.info(f"Приклад ключів: {list(ds[0].keys())}")
        
        # Збереження на диск, якщо потрібно
        if save_to_disk:
            if save_path is None:
                save_path = Path("data/meetingbank_dataset")
            else:
                save_path = Path(save_path)
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Збереження датасету до {save_path}")
            ds.save_to_disk(str(save_path))
            logger.info("Датасет збережено на диск!")
        
        return ds
        
    except Exception as e:
        logger.error(f"Помилка при завантаженні датасету: {e}")
        raise


if __name__ == "__main__":
    # Завантаження датасету
    ds = load_meetingbank_dataset(
        dataset_name="lytang/MeetingBank-transcript",
        save_to_disk=False  # Встановіть True, якщо хочете зберегти на диск
    )
    
    # Приклад використання
    print("\n" + "="*50)
    print("Приклад використання датасету:")
    print("="*50)
    
    if hasattr(ds, 'keys'):
        # Якщо це DatasetDict (має train/test/val розділи)
        for split_name in ds.keys():
            print(f"\nРозділ: {split_name}")
            if len(ds[split_name]) > 0:
                example = ds[split_name][0]
                print(f"  Перший запис має ключі: {list(example.keys())}")
    else:
        # Якщо це простий Dataset
        if len(ds) > 0:
            example = ds[0]
            print(f"Перший запис має ключі: {list(example.keys())}")